import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _round_half_up(self, prec):
    """Rounds 5 up (away from 0)"""
    if self._int[prec] in '56789':
        return 1
    elif _all_zeros(self._int, prec):
        return 0
    else:
        return -1