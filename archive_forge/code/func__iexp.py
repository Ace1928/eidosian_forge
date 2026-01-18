import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _iexp(x, M, L=8):
    """Given integers x and M, M > 0, such that x/M is small in absolute
    value, compute an integer approximation to M*exp(x/M).  For 0 <=
    x/M <= 2.4, the absolute error in the result is bounded by 60 (and
    is usually much smaller)."""
    R = _nbits((x << L) // M)
    T = -int(-10 * len(str(M)) // (3 * L))
    y = _div_nearest(x, T)
    Mshift = M << R
    for i in range(T - 1, 0, -1):
        y = _div_nearest(x * (Mshift + y), Mshift * i)
    for k in range(R - 1, -1, -1):
        Mshift = M << k + 2
        y = _div_nearest(y * (y + Mshift), Mshift)
    return M + y