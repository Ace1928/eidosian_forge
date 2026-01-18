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
def _compute_affine_output_size(matrix: List[float], w: int, h: int) -> Tuple[int, int]:
    half_w = 0.5 * w
    half_h = 0.5 * h
    pts = torch.tensor([[-half_w, -half_h, 1.0], [-half_w, half_h, 1.0], [half_w, half_h, 1.0], [half_w, -half_h, 1.0]])
    theta = torch.tensor(matrix, dtype=torch.float).view(2, 3)
    new_pts = torch.matmul(pts, theta.T)
    min_vals, max_vals = new_pts.aminmax(dim=0)
    halfs = torch.tensor((half_w, half_h))
    min_vals.add_(halfs)
    max_vals.add_(halfs)
    tol = 0.0001
    inv_tol = 1.0 / tol
    cmax = max_vals.mul_(inv_tol).trunc_().mul_(tol).ceil_()
    cmin = min_vals.mul_(inv_tol).trunc_().mul_(tol).floor_()
    size = cmax.sub_(cmin)
    return (int(size[0]), int(size[1]))