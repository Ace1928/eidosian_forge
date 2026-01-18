import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _div_nearest(a, b):
    """Closest integer to a/b, a and b positive integers; rounds to even
    in the case of a tie.

    """
    q, r = divmod(a, b)
    return q + (2 * r + (q & 1) > b)