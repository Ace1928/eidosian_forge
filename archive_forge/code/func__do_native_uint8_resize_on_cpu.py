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
def _do_native_uint8_resize_on_cpu(interpolation: InterpolationMode) -> bool:
    if interpolation == InterpolationMode.BILINEAR:
        if torch._dynamo.is_compiling():
            return True
        else:
            return 'AVX2' in torch.backends.cpu.get_cpu_capability()
    return interpolation == InterpolationMode.BICUBIC