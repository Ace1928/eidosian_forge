import operator
import math
import cmath
def _sinpi_real(x):
    if x < 0:
        return -_sinpi_real(-x)
    n, r = divmod(x, 0.5)
    r *= pi
    n %= 4
    if n == 0:
        return math.sin(r)
    if n == 1:
        return math.cos(r)
    if n == 2:
        return -math.sin(r)
    if n == 3:
        return -math.cos(r)