import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _round_up(self, prec):
    """Rounds away from 0."""
    return -self._round_down(prec)