import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _round_down(self, prec):
    """Also known as round-towards-0, truncate."""
    if _all_zeros(self._int, prec):
        return 0
    else:
        return -1