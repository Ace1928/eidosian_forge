import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _islogical(self):
    """Return True if self is a logical operand.

        For being logical, it must be a finite number with a sign of 0,
        an exponent of 0, and a coefficient whose digits must all be
        either 0 or 1.
        """
    if self._sign != 0 or self._exp != 0:
        return False
    for dig in self._int:
        if dig not in '01':
            return False
    return True