import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
@classmethod
def _masked_matmul(cls, a, b, mask):
    if not (type(a) is torch.Tensor and type(b) is torch.Tensor):
        return NotImplemented
    b = b.transpose(-2, -1)
    assert b.is_contiguous()
    if _can_use_triton(a):
        res = mask.__sparse_dot_sdd(a, b)
    else:
        res = _sddmm(a, b, mask.__layout)
    return cls._wrap(res, mask)