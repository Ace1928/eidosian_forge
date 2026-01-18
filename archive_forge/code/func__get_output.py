import warnings
import numpy
import cupy
def _get_output(output, input, shape=None, complex_output=False):
    shape = input.shape if shape is None else shape
    if output is None:
        if complex_output:
            _dtype = cupy.promote_types(input.dtype, cupy.complex64)
        else:
            _dtype = input.dtype
        output = cupy.empty(shape, dtype=_dtype)
    elif isinstance(output, (type, cupy.dtype)):
        if complex_output and cupy.dtype(output).kind != 'c':
            warnings.warn('promoting specified output dtype to complex')
            output = cupy.promote_types(output, cupy.complex64)
        output = cupy.empty(shape, dtype=output)
    elif isinstance(output, str):
        output = numpy.sctypeDict[output]
        if complex_output and cupy.dtype(output).kind != 'c':
            raise RuntimeError('output must have complex dtype')
        output = cupy.empty(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError('output shape not correct')
    elif complex_output and output.dtype.kind != 'c':
        raise RuntimeError('output must have complex dtype')
    return output