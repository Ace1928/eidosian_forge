import warnings
import numpy
import cupy
def _is_integer_output(output, input):
    if output is None:
        return input.dtype.kind in 'iu'
    elif isinstance(output, cupy.ndarray):
        return output.dtype.kind in 'iu'
    return cupy.dtype(output).kind in 'iu'