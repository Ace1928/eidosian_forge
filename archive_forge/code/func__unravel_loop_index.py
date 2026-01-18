import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
def _unravel_loop_index(shape, uint_t='unsigned int'):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.
    """
    ndim = len(shape)
    code = [f'\n        {uint_t} in_coord[{ndim}];\n        {uint_t} s, t, idx = i;']
    for j in range(ndim - 1, 0, -1):
        code.append(f'\n        s = {shape[j]};\n        t = idx / s;\n        in_coord[{j}] = idx - t * s;\n        idx = t;')
    code.append('\n        in_coord[0] = idx;')
    return '\n'.join(code)