import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def _prewitt_or_sobel(input, axis, output, mode, cval, weights):
    axis = internal._normalize_axis_index(axis, input.ndim)

    def get(is_diff):
        return cupy.array([-1, 0, 1], dtype=weights.dtype) if is_diff else weights
    return _run_1d_correlates(input, [a == axis for a in range(input.ndim)], get, output, mode, cval)