import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _sqrt_nearest(n, a):
    """Closest integer to the square root of the positive integer n.  a is
    an initial approximation to the square root.  Any positive integer
    will do for a, but the closer a is to the square root of n the
    faster convergence will be.

    """
    if n <= 0 or a <= 0:
        raise ValueError('Both arguments to _sqrt_nearest should be positive.')
    b = 0
    while a != b:
        b, a = (a, a - -n // a >> 1)
    return a