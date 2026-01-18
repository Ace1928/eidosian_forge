import operator
import math
import cmath
def _erf_taylor(x):
    x2 = x * x
    s = t = x
    n = 1
    while abs(t) > 1e-17:
        t *= x2 / n
        s -= t / (n + n + 1)
        n += 1
        t *= x2 / n
        s += t / (n + n + 1)
        n += 1
    return 1.1283791670955126 * s