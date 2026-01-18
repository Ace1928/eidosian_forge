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
def _perspective_coefficients(startpoints: Optional[List[List[int]]], endpoints: Optional[List[List[int]]], coefficients: Optional[List[float]]) -> List[float]:
    if coefficients is not None:
        if startpoints is not None and endpoints is not None:
            raise ValueError("The startpoints/endpoints and the coefficients shouldn't be defined concurrently.")
        elif len(coefficients) != 8:
            raise ValueError('Argument coefficients should have 8 float values')
        return coefficients
    elif startpoints is not None and endpoints is not None:
        return _get_perspective_coeffs(startpoints, endpoints)
    else:
        raise ValueError('Either the startpoints/endpoints or the coefficients must have non `None` values.')