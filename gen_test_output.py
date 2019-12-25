import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
import numpy as np


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=None):
    
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    test_path = '../brainwash-yolo/test.txt'  # path to test images

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size, 16]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    model.eval()
    with open('brainwash_test_output.txt', 'w') as f:
        for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='')):
            targets = targets.to(device)
            imgs = imgs.to(device)
            _, _, height, width = imgs.shape  # batch size, channels, height, width

            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs

            # Run NMS
            output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
            print('path: ', paths)
            print('output: ', output)
            f.write("%s;" % (paths[0]))
            if output[0] is None:
                f.write(";\n")
                continue
            output = output[0].cpu().numpy()
            print('shape: ', output.shape)
            for i in range(0, output.shape[0]):
                for j in range(0, output.shape[1]):
                    f.write("%f" % (output[i, j]))
                    if j < output.shape[1] - 1:
                        f.write(",")
                f.write(";")
            f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/brainwash.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='1', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.iou_thres,
             opt.conf_thres,
             opt.nms_thres,
             opt.save_json)
