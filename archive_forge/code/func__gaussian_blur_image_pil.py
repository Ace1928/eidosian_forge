import math
from typing import List, Optional
import PIL.Image
import torch
from torch.nn.functional import conv2d, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms._functional_tensor import _max_value
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(gaussian_blur, PIL.Image.Image)
def _gaussian_blur_image_pil(image: PIL.Image.Image, kernel_size: List[int], sigma: Optional[List[float]]=None) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = gaussian_blur_image(t_img, kernel_size=kernel_size, sigma=sigma)
    return to_pil_image(output, mode=image.mode)