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
@_register_five_ten_crop_kernel_internal(five_crop, PIL.Image.Image)
def _five_crop_image_pil(image: PIL.Image.Image, size: List[int]) -> Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    crop_height, crop_width = _parse_five_crop_size(size)
    image_height, image_width = _get_size_image_pil(image)
    if crop_width > image_width or crop_height > image_height:
        raise ValueError(f'Requested crop size {size} is bigger than input size {(image_height, image_width)}')
    tl = _crop_image_pil(image, 0, 0, crop_height, crop_width)
    tr = _crop_image_pil(image, 0, image_width - crop_width, crop_height, crop_width)
    bl = _crop_image_pil(image, image_height - crop_height, 0, crop_height, crop_width)
    br = _crop_image_pil(image, image_height - crop_height, image_width - crop_width, crop_height, crop_width)
    center = _center_crop_image_pil(image, [crop_height, crop_width])
    return (tl, tr, bl, br, center)