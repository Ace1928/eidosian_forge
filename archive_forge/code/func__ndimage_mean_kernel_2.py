import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _ndimage_mean_kernel_2(input, labels, index, batch_size=4, return_count=False):
    sum_val = cupy.empty_like(index, dtype=cupy.float64)
    count = cupy.empty_like(index, dtype=cupy.uint64)
    for i in range(0, index.size, batch_size):
        matched = labels == index[i:i + batch_size].reshape((-1,) + (1,) * input.ndim)
        mean_axes = tuple(range(1, 1 + input.ndim))
        count[i:i + batch_size] = matched.sum(axis=mean_axes)
        sum_val[i:i + batch_size] = cupy.where(matched, input, 0).sum(axis=mean_axes)
    if return_count:
        return (sum_val / count, count)
    return sum_val / count