import math
import operator
import warnings
import numpy
import numpy as np
from numpy import (atleast_1d, poly, polyval, roots, real, asarray,
from numpy.polynomial.polynomial import polyval as npp_polyval
from numpy.polynomial.polynomial import polyvalfromroots
from scipy import special, optimize, fft as sp_fft
from scipy.special import comb
from scipy._lib._util import float_factorial
def _bessel_zeros(N):
    """
    Find zeros of ordinary Bessel polynomial of order `N`, by root-finding of
    modified Bessel function of the second kind
    """
    if N == 0:
        return asarray([])
    x0 = _campos_zeros(N)

    def f(x):
        return special.kve(N + 0.5, 1 / x)

    def fp(x):
        return special.kve(N - 0.5, 1 / x) / (2 * x ** 2) - special.kve(N + 0.5, 1 / x) / x ** 2 + special.kve(N + 1.5, 1 / x) / (2 * x ** 2)
    x = _aberth(f, fp, x0)
    for i in range(len(x)):
        x[i] = optimize.newton(f, x[i], fp, tol=1e-15)
    x = np.mean((x, x[::-1].conj()), 0)
    if abs(np.sum(x) + 1) > 1e-15:
        raise RuntimeError('Generated zeros are inaccurate')
    return x