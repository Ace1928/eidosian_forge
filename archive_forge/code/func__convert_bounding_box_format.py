from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
def _convert_bounding_box_format(bounding_boxes: torch.Tensor, old_format: BoundingBoxFormat, new_format: BoundingBoxFormat, inplace: bool=False) -> torch.Tensor:
    if new_format == old_format:
        return bounding_boxes
    if old_format == BoundingBoxFormat.XYWH:
        bounding_boxes = _xywh_to_xyxy(bounding_boxes, inplace)
    elif old_format == BoundingBoxFormat.CXCYWH:
        bounding_boxes = _cxcywh_to_xyxy(bounding_boxes, inplace)
    if new_format == BoundingBoxFormat.XYWH:
        bounding_boxes = _xyxy_to_xywh(bounding_boxes, inplace)
    elif new_format == BoundingBoxFormat.CXCYWH:
        bounding_boxes = _xyxy_to_cxcywh(bounding_boxes, inplace)
    return bounding_boxes