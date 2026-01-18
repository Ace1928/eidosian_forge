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
def _get_spline_output(input, output):
    """Create workspace array, temp, and the final dtype for the output.

    Differs from SciPy by not always forcing the internal floating point dtype
    to be double precision.
    """
    complex_data = input.dtype.kind == 'c'
    if complex_data:
        min_float_dtype = cupy.complex64
    else:
        min_float_dtype = cupy.float32
    if isinstance(output, cupy.ndarray):
        if complex_data and output.dtype.kind != 'c':
            raise ValueError('output must have complex dtype for complex inputs')
        float_dtype = cupy.promote_types(output.dtype, min_float_dtype)
        output_dtype = output.dtype
    else:
        if output is None:
            output = output_dtype = input.dtype
        else:
            output_dtype = cupy.dtype(output)
        float_dtype = cupy.promote_types(output, min_float_dtype)
    if isinstance(output, cupy.ndarray) and output.dtype == float_dtype == output_dtype and output.flags.c_contiguous:
        if output is not input:
            _core.elementwise_copy(input, output)
        temp = output
    else:
        temp = input.astype(float_dtype, copy=False)
        temp = cupy.ascontiguousarray(temp)
        if cupy.shares_memory(temp, input, 'MAY_SHARE_BOUNDS'):
            temp = temp.copy()
    return (temp, float_dtype, output_dtype)