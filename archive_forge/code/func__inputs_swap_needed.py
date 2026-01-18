import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    if mode != 'valid' or not shape1:
        return False
    if axes is None:
        axes = tuple(range(len(shape1)))
    not_ok1 = any((shape1[i] < shape2[i] for i in axes))
    not_ok2 = any((shape1[i] > shape2[i] for i in axes))
    if not_ok1 and not_ok2:
        raise ValueError('For "valid" mode, one must be at least as large as the other in every dimension')
    return not_ok1