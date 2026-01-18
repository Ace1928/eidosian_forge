import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _check_nans(self, other=None, context=None):
    """Returns whether the number is not actually one.

        if self, other are sNaN, signal
        if self, other are NaN return nan
        return 0

        Done before operations.
        """
    self_is_nan = self._isnan()
    if other is None:
        other_is_nan = False
    else:
        other_is_nan = other._isnan()
    if self_is_nan or other_is_nan:
        if context is None:
            context = getcontext()
        if self_is_nan == 2:
            return context._raise_error(InvalidOperation, 'sNaN', self)
        if other_is_nan == 2:
            return context._raise_error(InvalidOperation, 'sNaN', other)
        if self_is_nan:
            return self._fix_nan(context)
        return other._fix_nan(context)
    return 0