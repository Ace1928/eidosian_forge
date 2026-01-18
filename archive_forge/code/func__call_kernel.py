import warnings
import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
def _call_kernel(kernel, input, weights, output, structure=None, weights_dtype=numpy.float64, structure_dtype=numpy.float64):
    """
    Calls a constructed ElementwiseKernel. The kernel must take an input image,
    an optional array of weights, an optional array for the structure, and an
    output array.

    weights and structure can be given as None (structure defaults to None) in
    which case they are not passed to the kernel at all. If the output is given
    as None then it will be allocated in this function.

    This function deals with making sure that the weights and structure are
    contiguous and float64 (or bool for weights that are footprints)*, that the
    output is allocated and appriopately shaped. This also deals with the
    situation that the input and output arrays overlap in memory.

    * weights is always cast to float64 or bool in order to get an output
    compatible with SciPy, though float32 might be sufficient when input dtype
    is low precision. If weights_dtype is passed as weights.dtype then no
    dtype conversion will occur. The input and output are never converted.
    """
    args = [input]
    complex_output = input.dtype.kind == 'c'
    if weights is not None:
        weights = cupy.ascontiguousarray(weights, weights_dtype)
        complex_output = complex_output or weights.dtype.kind == 'c'
        args.append(weights)
    if structure is not None:
        structure = cupy.ascontiguousarray(structure, structure_dtype)
        args.append(structure)
    output = _util._get_output(output, input, None, complex_output)
    needs_temp = cupy.shares_memory(output, input, 'MAY_SHARE_BOUNDS')
    if needs_temp:
        output, temp = (_util._get_output(output.dtype, input), output)
    args.append(output)
    kernel(*args)
    if needs_temp:
        _core.elementwise_copy(temp, output)
        output = temp
    return output