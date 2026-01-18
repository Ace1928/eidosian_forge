import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def _pad_symmetric(img: Tensor, padding: List[int]) -> Tensor:
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or (padding[3] < 0):
        neg_min_padding = [-min(x, 0) for x in padding]
        crop_left, crop_right, crop_top, crop_bottom = neg_min_padding
        img = img[..., crop_top:img.shape[-2] - crop_bottom, crop_left:img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]
    in_sizes = img.size()
    _x_indices = [i for i in range(in_sizes[-1])]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]
    right_indices = [-(i + 1) for i in range(padding[1])]
    x_indices = torch.tensor(left_indices + _x_indices + right_indices, device=img.device)
    _y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = torch.tensor(top_indices + _y_indices + bottom_indices, device=img.device)
    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]]
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]]
    else:
        raise RuntimeError('Symmetric padding of N-D tensors are not supported yet')