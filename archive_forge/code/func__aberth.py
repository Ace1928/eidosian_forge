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
def _aberth(f, fp, x0, tol=1e-15, maxiter=50):
    """
    Given a function `f`, its first derivative `fp`, and a set of initial
    guesses `x0`, simultaneously find the roots of the polynomial using the
    Aberth-Ehrlich method.

    ``len(x0)`` should equal the number of roots of `f`.

    (This is not a complete implementation of Bini's algorithm.)
    """
    N = len(x0)
    x = array(x0, complex)
    beta = np.empty_like(x0)
    for iteration in range(maxiter):
        alpha = -f(x) / fp(x)
        for k in range(N):
            beta[k] = np.sum(1 / (x[k] - x[k + 1:]))
            beta[k] += np.sum(1 / (x[k] - x[:k]))
        x += alpha / (1 + alpha * beta)
        if not all(np.isfinite(x)):
            raise RuntimeError('Root-finding calculation failed')
        if all(abs(alpha) <= tol):
            break
    else:
        raise Exception('Zeros failed to converge')
    return x