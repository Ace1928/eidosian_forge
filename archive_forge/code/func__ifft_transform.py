from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _ifft_transform(sst: Tensor, dim: int) -> Tensor:
    """Wrapper of torch.fft.ifft with more flexibility on dimensions.

    Args:
        sst (Tensor):
            Input sst tensor (may have zeros) in frequency domain.
        dim (int):
            Which dimension to transform.
    Returns:
        (Tensor):
            A new, transformed dense tensor with real domain values.
    """
    assert sst.is_complex()
    orig_shape = None
    if dim is None:
        orig_shape = sst.shape
        sst = sst.reshape(-1)
        dim = -1
    ret = torch.fft.ifft(sst, dim=dim)
    if orig_shape is not None:
        ret = ret.reshape(orig_shape)
    return ret