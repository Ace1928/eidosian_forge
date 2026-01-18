import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _convert_other(other, raiseit=False, allow_float=False):
    """Convert other to Decimal.

    Verifies that it's ok to use in an implicit construction.
    If allow_float is true, allow conversion from float;  this
    is used in the comparison methods (__eq__ and friends).

    """
    if isinstance(other, Decimal):
        return other
    if isinstance(other, int):
        return Decimal(other)
    if allow_float and isinstance(other, float):
        return Decimal.from_float(other)
    if raiseit:
        raise TypeError('Unable to convert %s to Decimal' % other)
    return NotImplemented