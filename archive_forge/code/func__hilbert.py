from functools import lru_cache
from math import ceil, pi
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn.functional import pad
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
def _hilbert(x: Tensor, n: Optional[int]=None) -> Tensor:
    if x.is_complex():
        raise ValueError('x must be real.')
    if n is None:
        n = x.shape[-1]
        if n % 16:
            n = ceil(n / 16) * 16
    if n <= 0:
        raise ValueError('N must be positive.')
    x_fft = torch.fft.fft(x, n=n, dim=-1)
    h = torch.zeros(n, dtype=x.dtype, device=x.device, requires_grad=False)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    y = torch.fft.ifft(x_fft * h, dim=-1)
    return y[..., :x.shape[-1]]