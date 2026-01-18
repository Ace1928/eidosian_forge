import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
@cupy._util.memoize(for_each_device=True)
def _get_shift_kernel(ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, nprepad=0):
    in_params = 'raw X x, raw W shift'
    out_params = 'Y y'
    operation, name = _generate_interp_custom(coord_func=_get_coord_shift, ndim=ndim, large_int=large_int, yshape=yshape, mode=mode, cval=cval, order=order, name='shift', integer_output=integer_output, nprepad=nprepad)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, preamble=math_constants_preamble)