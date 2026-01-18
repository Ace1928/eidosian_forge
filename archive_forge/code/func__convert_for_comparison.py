import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _convert_for_comparison(self, other, equality_op=False):
    """Given a Decimal instance self and a Python object other, return
    a pair (s, o) of Decimal instances such that "s op o" is
    equivalent to "self op other" for any of the 6 comparison
    operators "op".

    """
    if isinstance(other, Decimal):
        return (self, other)
    if isinstance(other, _numbers.Rational):
        if not self._is_special:
            self = _dec_from_triple(self._sign, str(int(self._int) * other.denominator), self._exp)
        return (self, Decimal(other.numerator))
    if equality_op and isinstance(other, _numbers.Complex) and (other.imag == 0):
        other = other.real
    if isinstance(other, float):
        context = getcontext()
        if equality_op:
            context.flags[FloatOperation] = 1
        else:
            context._raise_error(FloatOperation, 'strict semantics for mixing floats and Decimals are enabled')
        return (self, Decimal.from_float(other))
    return (NotImplemented, NotImplemented)