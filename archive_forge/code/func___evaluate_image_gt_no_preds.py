import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def __evaluate_image_gt_no_preds(self, gt: Tensor, gt_label_mask: Tensor, area_range: Tuple[int, int], num_iou_thrs: int) -> Dict[str, Any]:
    """Evaluate images with a ground truth but no predictions."""
    gt = [gt[i] for i in gt_label_mask]
    num_gt = len(gt)
    areas = compute_area(gt, iou_type=self.iou_type).to(self.device)
    ignore_area = (areas < area_range[0]) | (areas > area_range[1])
    gt_ignore, _ = torch.sort(ignore_area.to(torch.uint8))
    gt_ignore = gt_ignore.to(torch.bool)
    num_det = 0
    det_ignore = torch.zeros((num_iou_thrs, num_det), dtype=torch.bool, device=self.device)
    return {'dtMatches': torch.zeros((num_iou_thrs, num_det), dtype=torch.bool, device=self.device), 'gtMatches': torch.zeros((num_iou_thrs, num_gt), dtype=torch.bool, device=self.device), 'dtScores': torch.zeros(num_det, dtype=torch.float32, device=self.device), 'gtIgnore': gt_ignore, 'dtIgnore': det_ignore}