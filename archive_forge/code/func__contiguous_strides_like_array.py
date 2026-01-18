from contextlib import contextmanager
import numpy as np
from_record_like = None
def _contiguous_strides_like_array(ary):
    """
    Given an array, compute strides for a new contiguous array of the same
    shape.
    """
    if ary.flags['C_CONTIGUOUS'] or ary.flags['F_CONTIGUOUS'] or ary.ndim <= 1:
        return None
    strideperm = [x for x in enumerate(ary.strides)]
    strideperm.sort(key=lambda x: x[1])
    strides = [0] * len(ary.strides)
    stride = ary.dtype.itemsize
    for i_perm, _ in strideperm:
        strides[i_perm] = stride
        stride *= ary.shape[i_perm]
    return tuple(strides)