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
def _check_coordinates(coordinates, order, allow_float32=True):
    if coordinates.dtype.kind == 'f':
        if allow_float32:
            coord_dtype = cupy.promote_types(coordinates.dtype, cupy.float32)
        else:
            coord_dtype = cupy.promote_types(coordinates.dtype, cupy.float64)
        coordinates = coordinates.astype(coord_dtype, copy=False)
    elif coordinates.dtype.kind in 'iu':
        if order > 1:
            if allow_float32:
                coord_dtype = cupy.promote_types(coordinates.dtype, cupy.float32)
            else:
                coord_dtype = cupy.promote_types(coordinates.dtype, cupy.float64)
            coordinates = coordinates.astype(coord_dtype)
    else:
        raise ValueError('coordinates should have floating point dtype')
    if not coordinates.flags.c_contiguous:
        coordinates = cupy.ascontiguousarray(coordinates)
    return coordinates