import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _dlog(c, e, p):
    """Given integers c, e and p with c > 0, compute an integer
    approximation to 10**p * log(c*10**e), with an absolute error of
    at most 1.  Assumes that c*10**e is not exactly 1."""
    p += 2
    l = len(str(c))
    f = e + l - (e + l >= 1)
    if p > 0:
        k = e + p - f
        if k >= 0:
            c *= 10 ** k
        else:
            c = _div_nearest(c, 10 ** (-k))
        log_d = _ilog(c, 10 ** p)
    else:
        log_d = 0
    if f:
        extra = len(str(abs(f))) - 1
        if p + extra >= 0:
            f_log_ten = _div_nearest(f * _log10_digits(p + extra), 10 ** extra)
        else:
            f_log_ten = 0
    else:
        f_log_ten = 0
    return _div_nearest(f_log_ten + log_d, 100)