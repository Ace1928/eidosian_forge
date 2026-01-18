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
@_register_kernel_internal(rotate, PIL.Image.Image)
def _rotate_image_pil(image: PIL.Image.Image, angle: float, interpolation: Union[InterpolationMode, int]=InterpolationMode.NEAREST, expand: bool=False, center: Optional[List[float]]=None, fill: _FillTypeJIT=None) -> PIL.Image.Image:
    interpolation = _check_interpolation(interpolation)
    return _FP.rotate(image, angle, interpolation=pil_modes_mapping[interpolation], expand=expand, fill=fill, center=center)