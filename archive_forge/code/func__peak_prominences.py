import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit
def _peak_prominences(x, peaks, wlen=None, check=False):
    if check and cupy.any(cupy.logical_or(peaks < 0, peaks > x.shape[0] - 1)):
        raise ValueError('peaks are not a valid index')
    prominences = cupy.empty(peaks.shape[0], dtype=x.dtype)
    left_bases = cupy.empty(peaks.shape[0], dtype=cupy.int64)
    right_bases = cupy.empty(peaks.shape[0], dtype=cupy.int64)
    n = peaks.shape[0]
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz
    peak_prom_kernel = _get_module_func(PEAKS_MODULE, 'peak_prominences', x)
    peak_prom_kernel((n_blocks,), (block_sz,), (x.shape[0], n, x, peaks, wlen, prominences, left_bases, right_bases))
    return (prominences, left_bases, right_bases)