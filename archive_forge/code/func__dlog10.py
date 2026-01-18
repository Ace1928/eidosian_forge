import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _dlog10(c, e, p):
    """Given integers c, e and p with c > 0, p >= 0, compute an integer
    approximation to 10**p * log10(c*10**e), with an absolute error of
    at most 1.  Assumes that c*10**e is not exactly 1."""
    p += 2
    l = len(str(c))
    f = e + l - (e + l >= 1)
    if p > 0:
        M = 10 ** p
        k = e + p - f
        if k >= 0:
            c *= 10 ** k
        else:
            c = _div_nearest(c, 10 ** (-k))
        log_d = _ilog(c, M)
        log_10 = _log10_digits(p)
        log_d = _div_nearest(log_d * M, log_10)
        log_tenpower = f * M
    else:
        log_d = 0
        log_tenpower = _div_nearest(f, 10 ** (-p))
    return _div_nearest(log_tenpower + log_d, 100)