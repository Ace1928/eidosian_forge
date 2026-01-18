import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(erase, PIL.Image.Image)
def _erase_image_pil(image: PIL.Image.Image, i: int, j: int, h: int, w: int, v: torch.Tensor, inplace: bool=False) -> PIL.Image.Image:
    t_img = pil_to_tensor(image)
    output = erase_image(t_img, i=i, j=j, h=h, w=w, v=v, inplace=inplace)
    return to_pil_image(output, mode=image.mode)