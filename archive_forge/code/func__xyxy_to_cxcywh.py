from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
def _xyxy_to_cxcywh(xyxy: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        xyxy = xyxy.clone()
    xyxy[..., 2:].sub_(xyxy[..., :2])
    xyxy[..., :2].mul_(2).add_(xyxy[..., 2:]).div_(2, rounding_mode=None if xyxy.is_floating_point() else 'floor')
    return xyxy