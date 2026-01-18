import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def _quadratic_coeff(signal):
    zi = -3 + 2 * sqrt(2.0)
    K = len(signal)
    powers = zi ** arange(K)
    if K == 1:
        yplus = signal[0] + zi * add.reduce(powers * signal)
        output = zi / (zi - 1) * yplus
        return atleast_1d(output)
    state = lfiltic(1, r_[1, -zi], atleast_1d(add.reduce(powers * signal)))
    b = ones(1)
    a = r_[1, -zi]
    yplus, _ = lfilter(b, a, signal, zi=state)
    out_last = zi / (zi - 1) * yplus[K - 1]
    state = lfiltic(-zi, r_[1, -zi], atleast_1d(out_last))
    b = asarray([-zi])
    output, _ = lfilter(b, a, yplus[-2::-1], zi=state)
    output = r_[output[::-1], out_last]
    return output * 8.0