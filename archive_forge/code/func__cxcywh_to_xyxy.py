from typing import List, Optional, Tuple
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.tv_tensors import BoundingBoxFormat
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal, is_pure_tensor
def _cxcywh_to_xyxy(cxcywh: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        cxcywh = cxcywh.clone()
    half_wh = cxcywh[..., 2:].div(-2, rounding_mode=None if cxcywh.is_floating_point() else 'floor').abs_()
    cxcywh[..., :2].sub_(half_wh)
    cxcywh[..., 2:].add_(cxcywh[..., :2])
    return cxcywh