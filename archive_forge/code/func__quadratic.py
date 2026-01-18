import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def _quadratic(x):
    x = abs(asarray(x, dtype=float))
    b = BSpline.basis_element([-1.5, -0.5, 0.5, 1.5], extrapolate=False)
    out = b(x)
    out[(x < -1.5) | (x > 1.5)] = 0
    return out