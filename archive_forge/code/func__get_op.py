import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
def _get_op(self, op, dtype):
    if op not in _nccl_ops:
        raise RuntimeError(f'Unknown op {op} for NCCL')
    if dtype in 'FD' and op != nccl.NCCL_SUM:
        raise ValueError('Only nccl.SUM is supported for complex arrays')
    return _nccl_ops[op]