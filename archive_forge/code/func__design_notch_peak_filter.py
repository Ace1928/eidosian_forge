from math import pi
import math
import cupy
from cupyx.scipy.signal._iir_filter_conversions import (
from cupyx.scipy.signal._iir_filter_conversions import (
def _design_notch_peak_filter(w0, Q, ftype, fs=2.0):
    """
    Design notch or peak digital filter.

    Parameters
    ----------
    w0 : float
        Normalized frequency to remove from a signal. If `fs` is specified,
        this is in the same units as `fs`. By default, it is a normalized
        scalar that must satisfy  ``0 < w0 < 1``, with ``w0 = 1``
        corresponding to half of the sampling frequency.
    Q : float
        Quality factor. Dimensionless parameter that characterizes
        notch filter -3 dB bandwidth ``bw`` relative to its center
        frequency, ``Q = w0/bw``.
    ftype : str
        The type of IIR filter to design:

            - notch filter : ``notch``
            - peak filter  : ``peak``
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (``b``) and denominator (``a``) polynomials
        of the IIR filter.
    """
    w0 = float(w0)
    Q = float(Q)
    w0 = 2 * w0 / fs
    if w0 > 1.0 or w0 < 0.0:
        raise ValueError('w0 should be such that 0 < w0 < 1')
    bw = w0 / Q
    bw = bw * pi
    w0 = w0 * pi
    gb = 1 / math.sqrt(2)
    if ftype == 'notch':
        beta = math.sqrt(1.0 - gb ** 2.0) / gb * math.tan(bw / 2.0)
    elif ftype == 'peak':
        beta = gb / math.sqrt(1.0 - gb ** 2.0) * math.tan(bw / 2.0)
    else:
        raise ValueError('Unknown ftype.')
    gain = 1.0 / (1.0 + beta)
    if ftype == 'notch':
        b = [gain * x for x in (1.0, -2.0 * math.cos(w0), 1.0)]
    else:
        b = [(1.0 - gain) * x for x in (1.0, 0.0, -1.0)]
    a = [1.0, -2.0 * gain * math.cos(w0), 2.0 * gain - 1.0]
    a = cupy.asarray(a)
    b = cupy.asarray(b)
    return (b, a)