import cupy
import cupyx.scipy.fft
from cupy import _core
from cupy._core import _routines_math as _math
from cupy._core import fusion
from cupy.lib import stride_tricks
import numpy
def _dot_convolve(a1, a2, mode):
    offset = 0
    if a1.size < a2.size:
        a1, a2 = (a2, a1)
        offset = 1 - a2.size % 2
    dtype = cupy.result_type(a1, a2)
    n1, n2 = (a1.size, a2.size)
    a1 = a1.astype(dtype, copy=False)
    a2 = a2.astype(dtype, copy=False)
    if mode == 'full':
        out_size = n1 + n2 - 1
        a1 = cupy.pad(a1, n2 - 1)
    elif mode == 'same':
        out_size = n1
        pad_size = (n2 - 1) // 2 + offset
        a1 = cupy.pad(a1, (n2 - 1 - pad_size, pad_size))
    elif mode == 'valid':
        out_size = n1 - n2 + 1
    stride = a1.strides[0]
    a1 = stride_tricks.as_strided(a1, (out_size, n2), (stride, stride))
    output = _dot_kernel(a1, a2[::-1], axis=1)
    return output