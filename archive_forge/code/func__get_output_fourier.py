import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy import special
def _get_output_fourier(output, input, complex_only=False):
    types = [cupy.complex64, cupy.complex128]
    if not complex_only:
        types += [cupy.float32, cupy.float64]
    if output is None:
        if input.dtype in types:
            output = cupy.empty(input.shape, dtype=input.dtype)
        else:
            output = cupy.empty(input.shape, dtype=types[-1])
    elif type(output) is type:
        if output not in types:
            raise RuntimeError('output type not supported')
        output = cupy.empty(input.shape, dtype=output)
    elif output.shape != input.shape:
        raise RuntimeError('output shape not correct')
    return output