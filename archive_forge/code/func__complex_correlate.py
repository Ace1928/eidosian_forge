import functools
import itertools
import math
import numpy
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
from scipy import ndimage
from scipy.ndimage._filters import _gaussian_kernel1d
from . import types, float_types, complex_types
def _complex_correlate(array, kernel, real_dtype, convolve=False, mode='reflect', cval=0):
    """Utility to perform a reference complex-valued convolutions.

    When convolve==False, correlation is performed instead
    """
    array = numpy.asarray(array)
    kernel = numpy.asarray(kernel)
    complex_array = array.dtype.kind == 'c'
    complex_kernel = kernel.dtype.kind == 'c'
    if array.ndim == 1:
        func = ndimage.convolve1d if convolve else ndimage.correlate1d
    else:
        func = ndimage.convolve if convolve else ndimage.correlate
    if not convolve:
        kernel = kernel.conj()
    if complex_array and complex_kernel:
        output = func(array.real, kernel.real, output=real_dtype, mode=mode, cval=numpy.real(cval)) - func(array.imag, kernel.imag, output=real_dtype, mode=mode, cval=numpy.imag(cval)) + 1j * func(array.imag, kernel.real, output=real_dtype, mode=mode, cval=numpy.imag(cval)) + 1j * func(array.real, kernel.imag, output=real_dtype, mode=mode, cval=numpy.real(cval))
    elif complex_array:
        output = func(array.real, kernel, output=real_dtype, mode=mode, cval=numpy.real(cval)) + 1j * func(array.imag, kernel, output=real_dtype, mode=mode, cval=numpy.imag(cval))
    elif complex_kernel:
        output = func(array, kernel.real, output=real_dtype, mode=mode, cval=cval) + 1j * func(array, kernel.imag, output=real_dtype, mode=mode, cval=cval)
    return output