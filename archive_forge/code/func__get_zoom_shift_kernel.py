import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
@cupy._util.memoize(for_each_device=True)
def _get_zoom_shift_kernel(ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False, grid_mode=False, nprepad=0):
    in_params = 'raw X x, raw W shift, raw W zoom'
    out_params = 'Y y'
    if grid_mode:
        zoom_shift_func = _get_coord_zoom_and_shift_grid
    else:
        zoom_shift_func = _get_coord_zoom_and_shift
    operation, name = _generate_interp_custom(coord_func=zoom_shift_func, ndim=ndim, large_int=large_int, yshape=yshape, mode=mode, cval=cval, order=order, name='zoom_shift_grid' if grid_mode else 'zoom_shift', integer_output=integer_output, nprepad=nprepad)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name, preamble=math_constants_preamble)