import torch
from xformers.ops import masked_matmul
from xformers.sparse import _csr_ops
from xformers.sparse.utils import (
@classmethod
def _binary_op_slow(cls, func, arg0, arg1):
    v0, v1 = (arg0, arg1)
    if isinstance(arg0, cls):
        v0 = arg0.to_dense()
    if isinstance(arg1, cls):
        v1 = arg1.to_dense()
    out = func(v0, v1)
    return cls.from_dense(out)