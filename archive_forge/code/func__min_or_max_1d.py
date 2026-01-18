import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def _min_or_max_1d(input, size, axis=-1, output=None, mode='reflect', cval=0.0, origin=0, func='min'):
    ftprnt = cupy.ones(size, dtype=bool)
    ftprnt, origin = _filters_core._convert_1d_args(input.ndim, ftprnt, origin, axis)
    origins, int_type = _filters_core._check_nd_args(input, ftprnt, mode, origin, 'footprint')
    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    kernel = _get_min_or_max_kernel(mode, ftprnt.shape, func, offsets, float(cval), int_type, has_weights=False)
    return _filters_core._call_kernel(kernel, input, None, output, weights_dtype=bool)