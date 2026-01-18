import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def _pad_with_vector_fill(image: torch.Tensor, torch_padding: List[int], fill: List[float], padding_mode: str) -> torch.Tensor:
    if padding_mode != 'constant':
        raise ValueError(f"Padding mode '{padding_mode}' is not supported if fill is not scalar")
    output = _pad_with_scalar_fill(image, torch_padding, fill=0, padding_mode='constant')
    left, right, top, bottom = torch_padding
    fill = torch.tensor(fill, device=image.device).to(dtype=image.dtype).reshape(-1, 1, 1)
    if top > 0:
        output[..., :top, :] = fill
    if left > 0:
        output[..., :, :left] = fill
    if bottom > 0:
        output[..., -bottom:, :] = fill
    if right > 0:
        output[..., :, -right:] = fill
    return output