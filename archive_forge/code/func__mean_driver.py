import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _mean_driver(input, labels, index, return_count=False, use_kern=False):
    if use_kern:
        return _ndimage_mean_kernel_2(input, labels, index, return_count=return_count)
    out = cupy.zeros_like(index, cupy.float64)
    count = cupy.zeros_like(index, dtype=cupy.uint64)
    sum, count = _ndimage_mean_kernel(input, labels, index, index.size, out, count)
    if return_count:
        return (sum / count, count)
    return sum / count