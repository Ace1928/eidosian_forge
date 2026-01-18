from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align
from . import _utils as det_utils
@torch.jit._script_if_tracing
def _onnx_heatmaps_to_keypoints_loop(maps, rois, widths_ceil, heights_ceil, widths, heights, offset_x, offset_y, num_keypoints):
    xy_preds = torch.zeros((0, 3, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((0, int(num_keypoints)), dtype=torch.float32, device=maps.device)
    for i in range(int(rois.size(0))):
        xy_preds_i, end_scores_i = _onnx_heatmaps_to_keypoints(maps, maps[i], widths_ceil[i], heights_ceil[i], widths[i], heights[i], offset_x[i], offset_y[i])
        xy_preds = torch.cat((xy_preds.to(dtype=torch.float32), xy_preds_i.unsqueeze(0).to(dtype=torch.float32)), 0)
        end_scores = torch.cat((end_scores.to(dtype=torch.float32), end_scores_i.to(dtype=torch.float32).unsqueeze(0)), 0)
    return (xy_preds, end_scores)