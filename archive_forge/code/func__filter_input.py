import math
import warnings
import cupy
import numpy
from cupy import _core
from cupy._core import internal
from cupy.cuda import runtime
from cupyx import _texture
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _interp_kernels
from cupyx.scipy.ndimage import _spline_prefilter_core
def _filter_input(image, prefilter, mode, cval, order):
    """Perform spline prefiltering when needed.

    Spline orders > 1 need a prefiltering stage to preserve resolution.

    For boundary modes without analytical spline boundary conditions, some
    prepadding of the input with cupy.pad is used to maintain accuracy.
    ``npad`` is an integer corresponding to the amount of padding at each edge
    of the array.
    """
    if not prefilter or order < 2:
        return (cupy.ascontiguousarray(image), 0)
    padded, npad = _prepad_for_spline_filter(image, mode, cval)
    float_dtype = cupy.promote_types(image.dtype, cupy.float32)
    filtered = spline_filter(padded, order, output=float_dtype, mode=mode)
    return (cupy.ascontiguousarray(filtered), npad)