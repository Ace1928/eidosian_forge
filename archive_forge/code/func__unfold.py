import functools
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, conv3d, pad, unfold
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE
def _unfold(x: Tensor, kernel_size: Tuple[int, ...]) -> Tensor:
    """Unfold the input tensor to a matrix. Function supports 3d images e.g. (B, C, D, H, W).

    Inspired by:
    https://github.com/f-dangel/unfoldNd/blob/main/unfoldNd/unfold.py

    Args:
        x: Input tensor to be unfolded.
        kernel_size: The size of the sliding blocks in each dimension.

    """
    batch_size, channels = x.shape[:2]
    n = x.ndim - 2
    if n == 2:
        return unfold(x, kernel_size)
    kernel_size_numel = kernel_size[0] * kernel_size[1] * kernel_size[2]
    repeat = [channels, 1] + [1 for _ in kernel_size]
    weight = torch.eye(kernel_size_numel, device=x.device, dtype=x.dtype)
    weight = weight.reshape(kernel_size_numel, 1, *kernel_size).repeat(*repeat)
    unfold_x = conv3d(x, weight=weight, bias=None)
    return unfold_x.reshape(batch_size, channels * kernel_size_numel, -1)