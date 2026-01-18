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
def _falling_factorial(x, n):
    """
    Return the factorial of `x` to the `n` falling.

    This is defined as:

    .. math::   x^\\underline n = (x)_n = x (x-1) \\cdots (x-n+1)

    This can more efficiently calculate ratios of factorials, since:

    n!/m! == falling_factorial(n, n-m)

    where n >= m

    skipping the factors that cancel out

    the usual factorial n! == ff(n, n)
    """
    val = 1
    for k in range(x - n + 1, x + 1):
        val *= k
    return val