import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
@classmethod
def _bmm(cls, arg0, arg1):
    if not (isinstance(arg0, cls) and type(arg1) is torch.Tensor):
        return NotImplemented
    if _can_use_triton(arg1):
        res = arg0.__sparse_dot_dsd(arg0.__values, arg1)
    else:
        res = _spmm(arg1, arg0.__layout, arg0.__values)
    return res