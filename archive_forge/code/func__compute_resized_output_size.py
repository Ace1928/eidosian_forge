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
def _compute_resized_output_size(canvas_size: Tuple[int, int], size: List[int], max_size: Optional[int]=None) -> List[int]:
    if isinstance(size, int):
        size = [size]
    elif max_size is not None and len(size) != 1:
        raise ValueError('max_size should only be passed if size specifies the length of the smaller edge, i.e. size should be an int or a sequence of length 1 in torchscript mode.')
    return __compute_resized_output_size(canvas_size, size=size, max_size=max_size)