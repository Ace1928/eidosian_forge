import warnings
import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
def _run_1d_filters(filters, input, args, output, mode, cval, origin=0):
    """
    Runs a series of 1D filters forming an nd filter. The filters must be a
    list of callables that take input, arg, axis, output, mode, cval, origin.
    The args is a list of values that are passed for the arg value to the
    filter. Individual filters can be None causing that axis to be skipped.
    """
    output = _util._get_output(output, input)
    modes = _util._fix_sequence_arg(mode, input.ndim, 'mode', _util._check_mode)
    modes = ['grid-wrap' if m == 'wrap' else m for m in modes]
    origins = _util._fix_sequence_arg(origin, input.ndim, 'origin', int)
    n_filters = sum((filter is not None for filter in filters))
    if n_filters == 0:
        _core.elementwise_copy(input, output)
        return output
    temp = _util._get_output(output.dtype, input) if n_filters > 1 else None
    iterator = zip(filters, args, modes, origins)
    for axis, (fltr, arg, mode, origin) in enumerate(iterator):
        if fltr is not None:
            break
    if n_filters % 2 == 0:
        fltr(input, arg, axis, temp, mode, cval, origin)
        input = temp
    else:
        fltr(input, arg, axis, output, mode, cval, origin)
        input, output = (output, temp)
    for axis, (fltr, arg, mode, origin) in enumerate(iterator, start=axis + 1):
        if fltr is None:
            continue
        fltr(input, arg, axis, output, mode, cval, origin)
        input, output = (output, input)
    return input