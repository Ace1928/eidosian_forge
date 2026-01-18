import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class Rounded(DecimalException):
    """Number got rounded (not  necessarily changed during rounding).

    This occurs and signals rounded whenever the result of an operation is
    rounded (that is, some zero or non-zero digits were discarded from the
    coefficient), or if an overflow or underflow condition occurs.  The
    result in all cases is unchanged.

    The rounded signal may be tested (or trapped) to determine if a given
    operation (or sequence of operations) caused a loss of precision.
    """