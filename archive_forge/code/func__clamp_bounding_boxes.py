from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
def _clamp_bounding_boxes(bounding_boxes: torch.Tensor, format: BoundingBoxFormat, canvas_size: Tuple[int, int]) -> torch.Tensor:
    in_dtype = bounding_boxes.dtype
    bounding_boxes = bounding_boxes.clone() if bounding_boxes.is_floating_point() else bounding_boxes.float()
    xyxy_boxes = convert_bounding_box_format(bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXY, inplace=True)
    xyxy_boxes[..., 0::2].clamp_(min=0, max=canvas_size[1])
    xyxy_boxes[..., 1::2].clamp_(min=0, max=canvas_size[0])
    out_boxes = convert_bounding_box_format(xyxy_boxes, old_format=BoundingBoxFormat.XYXY, new_format=format, inplace=True)
    return out_boxes.to(in_dtype)