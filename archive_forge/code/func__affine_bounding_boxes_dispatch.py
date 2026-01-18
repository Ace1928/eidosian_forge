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
@_register_kernel_internal(affine, tv_tensors.BoundingBoxes, tv_tensor_wrapper=False)
def _affine_bounding_boxes_dispatch(inpt: tv_tensors.BoundingBoxes, angle: Union[int, float], translate: List[float], scale: float, shear: List[float], center: Optional[List[float]]=None, **kwargs) -> tv_tensors.BoundingBoxes:
    output = affine_bounding_boxes(inpt.as_subclass(torch.Tensor), format=inpt.format, canvas_size=inpt.canvas_size, angle=angle, translate=translate, scale=scale, shear=shear, center=center)
    return tv_tensors.wrap(output, like=inpt)