import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _apply_conv_mode(full, s1, s2, mode, axes):
    if mode == 'full':
        return cupy.ascontiguousarray(full)
    if mode == 'valid':
        s1 = [full.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(full.ndim)]
    starts = [(cur - new) // 2 for cur, new in zip(full.shape, s1)]
    slices = tuple((slice(start, start + length) for start, length in zip(starts, s1)))
    return cupy.ascontiguousarray(full[slices])