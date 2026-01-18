import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _rshift_nearest(x, shift):
    """Given an integer x and a nonnegative integer shift, return closest
    integer to x / 2**shift; use round-to-even in case of a tie.

    """
    b, q = (1 << shift, x >> shift)
    return q + (2 * (x & b - 1) + (q & 1) > b)