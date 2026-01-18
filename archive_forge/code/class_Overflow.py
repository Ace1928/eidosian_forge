import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class Overflow(Inexact, Rounded):
    """Numerical overflow.

    This occurs and signals overflow if the adjusted exponent of a result
    (from a conversion or from an operation that is not an attempt to divide
    by zero), after rounding, would be greater than the largest value that
    can be handled by the implementation (the value Emax).

    The result depends on the rounding mode:

    For round-half-up and round-half-even (and for round-half-down and
    round-up, if implemented), the result of the operation is [sign,inf],
    where sign is the sign of the intermediate result.  For round-down, the
    result is the largest finite number that can be represented in the
    current precision, with the sign of the intermediate result.  For
    round-ceiling, the result is the same as for round-down if the sign of
    the intermediate result is 1, or is [0,inf] otherwise.  For round-floor,
    the result is the same as for round-down if the sign of the intermediate
    result is 0, or is [1,inf] otherwise.  In all cases, Inexact and Rounded
    will also be raised.
    """

    def handle(self, context, sign, *args):
        if context.rounding in (ROUND_HALF_UP, ROUND_HALF_EVEN, ROUND_HALF_DOWN, ROUND_UP):
            return _SignedInfinity[sign]
        if sign == 0:
            if context.rounding == ROUND_CEILING:
                return _SignedInfinity[sign]
            return _dec_from_triple(sign, '9' * context.prec, context.Emax - context.prec + 1)
        if sign == 1:
            if context.rounding == ROUND_FLOOR:
                return _SignedInfinity[sign]
            return _dec_from_triple(sign, '9' * context.prec, context.Emax - context.prec + 1)