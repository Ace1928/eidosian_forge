from typing import Tuple
import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._box_convert import _box_cxcywh_to_xyxy, _box_xywh_to_xyxy, _box_xyxy_to_cxcywh, _box_xyxy_to_xywh
from ._utils import _upcast
@torch.jit._script_if_tracing
def _batched_nms_vanilla(boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float) -> Tensor:
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]