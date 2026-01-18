import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class Clamped(DecimalException):
    """Exponent of a 0 changed to fit bounds.

    This occurs and signals clamped if the exponent of a result has been
    altered in order to fit the constraints of a specific concrete
    representation.  This may occur when the exponent of a zero result would
    be outside the bounds of a representation, or when a large normal
    number would have an encoded exponent that cannot be represented.  In
    this latter case, the exponent is reduced to fit and the corresponding
    number of zero digits are appended to the coefficient ("fold-down").
    """