from collections.abc import Iterable
import numbers
import warnings
import numpy
import operator
from scipy._lib._util import normalize_axis_index
from . import _ni_support
from . import _nd_image
from . import _ni_docstrings
def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input and complex_weights:
        func(input.real, weights.real, output=output.real, cval=numpy.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None, cval=numpy.imag(cval), **kwargs)
        func(input.real, weights.imag, output=output.imag, cval=numpy.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None, cval=numpy.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=numpy.real(cval), **kwargs)
        func(input.imag, weights, output=output.imag, cval=numpy.imag(cval), **kwargs)
    else:
        if numpy.iscomplexobj(cval):
            raise ValueError('Cannot provide a complex-valued cval when the input is real.')
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output