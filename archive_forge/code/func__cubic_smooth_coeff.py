import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def _cubic_smooth_coeff(signal, lamb):
    rho, omega = _coeff_smooth(lamb)
    cs = 1 - 2 * rho * cos(omega) + rho * rho
    K = len(signal)
    k = arange(K)
    zi_2 = _hc(0, cs, rho, omega) * signal[0] + add.reduce(_hc(k + 1, cs, rho, omega) * signal)
    zi_1 = _hc(0, cs, rho, omega) * signal[0] + _hc(1, cs, rho, omega) * signal[1] + add.reduce(_hc(k + 2, cs, rho, omega) * signal)
    zi = lfiltic(cs, r_[1, -2 * rho * cos(omega), rho * rho], r_[zi_1, zi_2])
    zi = zi.reshape(1, -1)
    sos = r_[cs, 0, 0, 1, -2 * rho * cos(omega), rho * rho]
    sos = sos.reshape(1, -1)
    yp, _ = sosfilt(sos, signal[2:], zi=zi)
    yp = r_[zi_2, zi_1, yp]
    zi_2 = add.reduce((_hs(k, cs, rho, omega) + _hs(k + 1, cs, rho, omega)) * signal[::-1])
    zi_1 = add.reduce((_hs(k - 1, cs, rho, omega) + _hs(k + 2, cs, rho, omega)) * signal[::-1])
    zi = lfiltic(cs, r_[1, -2 * rho * cos(omega), rho * rho], r_[zi_1, zi_2])
    zi = zi.reshape(1, -1)
    y, _ = sosfilt(sos, yp[-3::-1], zi=zi)
    y = r_[y[::-1], zi_1, zi_2]
    return y