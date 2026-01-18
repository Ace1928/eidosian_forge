import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
def _wraps_polyroutine(func):

    def _get_coeffs(x):
        if isinstance(x, cupy.poly1d):
            return x._coeffs
        if cupy.isscalar(x):
            return cupy.atleast_1d(x)
        if isinstance(x, cupy.ndarray):
            x = cupy.atleast_1d(x)
            if x.ndim == 1:
                return x
            raise ValueError('Multidimensional inputs are not supported')
        raise TypeError('Unsupported type')

    def wrapper(*args):
        coeffs = [_get_coeffs(x) for x in args]
        out = func(*coeffs)
        if all((not isinstance(x, cupy.poly1d) for x in args)):
            return out
        if isinstance(out, cupy.ndarray):
            return cupy.poly1d(out)
        if isinstance(out, tuple):
            return tuple([cupy.poly1d(x) for x in out])
        assert False
    return functools.update_wrapper(wrapper, func)