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
@_register_kernel_internal(resize, PIL.Image.Image)
def __resize_image_pil_dispatch(image: PIL.Image.Image, size: Union[Sequence[int], int], interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, max_size: Optional[int]=None, antialias: Optional[Union[str, bool]]='warn') -> PIL.Image.Image:
    if antialias is False:
        warnings.warn('Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.')
    return _resize_image_pil(image, size=size, interpolation=interpolation, max_size=max_size)