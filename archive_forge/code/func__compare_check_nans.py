import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _compare_check_nans(self, other, context):
    """Version of _check_nans used for the signaling comparisons
        compare_signal, __le__, __lt__, __ge__, __gt__.

        Signal InvalidOperation if either self or other is a (quiet
        or signaling) NaN.  Signaling NaNs take precedence over quiet
        NaNs.

        Return 0 if neither operand is a NaN.

        """
    if context is None:
        context = getcontext()
    if self._is_special or other._is_special:
        if self.is_snan():
            return context._raise_error(InvalidOperation, 'comparison involving sNaN', self)
        elif other.is_snan():
            return context._raise_error(InvalidOperation, 'comparison involving sNaN', other)
        elif self.is_qnan():
            return context._raise_error(InvalidOperation, 'comparison involving NaN', self)
        elif other.is_qnan():
            return context._raise_error(InvalidOperation, 'comparison involving NaN', other)
    return 0