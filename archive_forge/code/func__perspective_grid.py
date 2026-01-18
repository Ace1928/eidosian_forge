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
def _perspective_grid(coeffs: List[float], ow: int, oh: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    theta1 = torch.tensor([[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]], dtype=dtype, device=device)
    theta2 = torch.tensor([[[coeffs[6], coeffs[7], 1.0], [coeffs[6], coeffs[7], 1.0]]], dtype=dtype, device=device)
    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=dtype, device=device)
    x_grid = torch.linspace(d, ow + d - 1.0, steps=ow, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh + d - 1.0, steps=oh, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)
    rescaled_theta1 = theta1.transpose(1, 2).div_(torch.tensor([0.5 * ow, 0.5 * oh], dtype=dtype, device=device))
    shape = (1, oh * ow, 3)
    output_grid1 = base_grid.view(shape).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(shape).bmm(theta2.transpose(1, 2))
    output_grid = output_grid1.div_(output_grid2).sub_(1.0)
    return output_grid.view(1, oh, ow, 2)