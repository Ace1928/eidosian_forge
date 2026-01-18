from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
def _mul_add(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Element-wise multiplication of two complex Tensors described
    through their real and imaginary parts.
    The result is added to the `out` tensor"""
    target_shape = torch.Size([max(sa, sb) for sa, sb in zip(a.shape, b.shape)])
    if out is None or out.shape != target_shape:
        out = torch.zeros(target_shape, dtype=a.dtype, device=a.device)
    if out is a:
        real_a = a[..., 0]
        out[..., 0] = out[..., 0] + (real_a * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (real_a * b[..., 1] + a[..., 1] * b[..., 0])
    else:
        out[..., 0] = out[..., 0] + (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        out[..., 1] = out[..., 1] + (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])
    return out