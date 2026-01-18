import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _dec_from_triple(sign, coefficient, exponent, special=False):
    """Create a decimal instance directly, without any validation,
    normalization (e.g. removal of leading zeros) or argument
    conversion.

    This function is for *internal use only*.
    """
    self = object.__new__(Decimal)
    self._sign = sign
    self._int = coefficient
    self._exp = exponent
    self._is_special = special
    return self