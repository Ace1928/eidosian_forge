from typing import List
import PIL.Image
import torch
from torch.nn.functional import conv2d
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _max_value
from torchvision.utils import _log_api_usage_once
from ._misc import _num_value_bits, to_dtype_image
from ._type_conversion import pil_to_tensor, to_pil_image
from ._utils import _get_kernel, _register_kernel_internal
def _blend(image1: torch.Tensor, image2: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    fp = image1.is_floating_point()
    bound = _max_value(image1.dtype)
    output = image1.mul(ratio).add_(image2, alpha=1.0 - ratio).clamp_(0, bound)
    return output if fp else output.to(image1.dtype)