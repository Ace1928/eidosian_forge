from typing import Tuple
import torch
from ..utils import _log_api_usage_once
from ._utils import _loss_inter_union, _upcast_non_float
def _diou_iou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float=1e-07) -> Tuple[torch.Tensor, torch.Tensor]:
    intsct, union = _loss_inter_union(boxes1, boxes2)
    iou = intsct / (union + eps)
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diagonal_distance_squared = (xc2 - xc1) ** 2 + (yc2 - yc1) ** 2 + eps
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2
    loss = 1 - iou + centers_distance_squared / diagonal_distance_squared
    return (loss, iou)