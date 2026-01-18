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
def _rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r, g, _ = image.unbind(dim=-3)
    minc, maxc = torch.aminmax(image, dim=-3)
    eqc = maxc == minc
    channels_range = maxc - minc
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(dim=-3)
    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g
    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = gc.add_(4.0).sub_(rc).mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))
    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-3)