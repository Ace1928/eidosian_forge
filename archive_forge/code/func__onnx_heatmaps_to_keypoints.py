from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from . import _utils as det_utils
def _onnx_heatmaps_to_keypoints(maps, maps_i, roi_map_width, roi_map_height, widths_i, heights_i, offset_x_i, offset_y_i):
    num_keypoints = torch.scalar_tensor(maps.size(1), dtype=torch.int64)
    width_correction = widths_i / roi_map_width
    height_correction = heights_i / roi_map_height
    roi_map = F.interpolate(maps_i[:, None], size=(int(roi_map_height), int(roi_map_width)), mode='bicubic', align_corners=False)[:, 0]
    w = torch.scalar_tensor(roi_map.size(2), dtype=torch.int64)
    pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)
    x_int = pos % w
    y_int = (pos - x_int) // w
    x = (torch.tensor(0.5, dtype=torch.float32) + x_int.to(dtype=torch.float32)) * width_correction.to(dtype=torch.float32)
    y = (torch.tensor(0.5, dtype=torch.float32) + y_int.to(dtype=torch.float32)) * height_correction.to(dtype=torch.float32)
    xy_preds_i_0 = x + offset_x_i.to(dtype=torch.float32)
    xy_preds_i_1 = y + offset_y_i.to(dtype=torch.float32)
    xy_preds_i_2 = torch.ones(xy_preds_i_1.shape, dtype=torch.float32)
    xy_preds_i = torch.stack([xy_preds_i_0.to(dtype=torch.float32), xy_preds_i_1.to(dtype=torch.float32), xy_preds_i_2.to(dtype=torch.float32)], 0)
    base = num_keypoints * num_keypoints + num_keypoints + 1
    ind = torch.arange(num_keypoints)
    ind = ind.to(dtype=torch.int64) * base
    end_scores_i = roi_map.index_select(1, y_int.to(dtype=torch.int64)).index_select(2, x_int.to(dtype=torch.int64)).view(-1).index_select(0, ind.to(dtype=torch.int64))
    return (xy_preds_i, end_scores_i)