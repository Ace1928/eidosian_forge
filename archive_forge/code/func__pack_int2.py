import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _pack_int2(arr, marker=-32768, int_dtype=cupy.int16):
    if arr.ndim != 2:
        raise ValueError('only 2d arr suppported')
    int2_dtype = cupy.dtype({'names': ['x', 'y'], 'formats': [int_dtype] * 2})
    out = cupy.zeros(arr.shape + (2,), dtype=int_dtype)
    assert out.size == 2 * arr.size
    pack_kernel = _get_pack_kernel(int_type='short' if int_dtype == cupy.int16 else 'int', marker=marker)
    pack_kernel(arr, out, size=arr.size)
    out = cupy.squeeze(out.view(int2_dtype))
    return out