import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class DecimalException(ArithmeticError):
    """Base exception class.

    Used exceptions derive from this.
    If an exception derives from another exception besides this (such as
    Underflow (Inexact, Rounded, Subnormal) that indicates that it is only
    called if the others are present.  This isn't actually used for
    anything, though.

    handle  -- Called when context._raise_error is called and the
               trap_enabler is not set.  First argument is self, second is the
               context.  More arguments can be given, those being after
               the explanation in _raise_error (For example,
               context._raise_error(NewError, '(-x)!', self._sign) would
               call NewError().handle(context, self._sign).)

    To define a new exception, it should be sufficient to have it derive
    from DecimalException.
    """

    def handle(self, context, *args):
        pass