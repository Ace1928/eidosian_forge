import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _log10_lb(c, correction={'1': 100, '2': 70, '3': 53, '4': 40, '5': 31, '6': 23, '7': 16, '8': 10, '9': 5}):
    """Compute a lower bound for 100*log10(c) for a positive integer c."""
    if c <= 0:
        raise ValueError('The argument to _log10_lb should be nonnegative.')
    str_c = str(c)
    return 100 * len(str_c) - correction[str_c[0]]