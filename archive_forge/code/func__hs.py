import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def _hs(k, cs, rho, omega):
    c0 = cs * cs * (1 + rho * rho) / (1 - rho * rho) / (1 - 2 * rho * rho * cos(2 * omega) + rho ** 4)
    gamma = (1 - rho * rho) / (1 + rho * rho) / tan(omega)
    ak = abs(k)
    return c0 * rho ** ak * (cos(omega * ak) + gamma * sin(omega * ak))