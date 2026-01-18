import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def _bspline_piecefunctions(order):
    """Returns the function defined over the left-side pieces for a bspline of
    a given order.

    The 0th piece is the first one less than 0. The last piece is a function
    identical to 0 (returned as the constant 0). (There are order//2 + 2 total
    pieces).

    Also returns the condition functions that when evaluated return boolean
    arrays for use with `numpy.piecewise`.
    """
    try:
        return _splinefunc_cache[order]
    except KeyError:
        pass

    def condfuncgen(num, val1, val2):
        if num == 0:
            return lambda x: logical_and(less_equal(x, val1), greater_equal(x, val2))
        elif num == 2:
            return lambda x: less_equal(x, val2)
        else:
            return lambda x: logical_and(less(x, val1), greater_equal(x, val2))
    last = order // 2 + 2
    if order % 2:
        startbound = -1.0
    else:
        startbound = -0.5
    condfuncs = [condfuncgen(0, 0, startbound)]
    bound = startbound
    for num in range(1, last - 1):
        condfuncs.append(condfuncgen(1, bound, bound - 1))
        bound = bound - 1
    condfuncs.append(condfuncgen(2, 0, -(order + 1) / 2.0))
    fval = float_factorial(order)

    def piecefuncgen(num):
        Mk = order // 2 - num
        if Mk < 0:
            return 0
        coeffs = [(1 - 2 * (k % 2)) * float(comb(order + 1, k, exact=1)) / fval for k in range(Mk + 1)]
        shifts = [-bound - k for k in range(Mk + 1)]

        def thefunc(x):
            res = 0.0
            for k in range(Mk + 1):
                res += coeffs[k] * (x + shifts[k]) ** order
            return res
        return thefunc
    funclist = [piecefuncgen(k) for k in range(last)]
    _splinefunc_cache[order] = (funclist, condfuncs)
    return (funclist, condfuncs)