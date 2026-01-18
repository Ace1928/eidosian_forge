import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _power_modulo(self, other, modulo, context=None):
    """Three argument version of __pow__"""
    other = _convert_other(other)
    if other is NotImplemented:
        return other
    modulo = _convert_other(modulo)
    if modulo is NotImplemented:
        return modulo
    if context is None:
        context = getcontext()
    self_is_nan = self._isnan()
    other_is_nan = other._isnan()
    modulo_is_nan = modulo._isnan()
    if self_is_nan or other_is_nan or modulo_is_nan:
        if self_is_nan == 2:
            return context._raise_error(InvalidOperation, 'sNaN', self)
        if other_is_nan == 2:
            return context._raise_error(InvalidOperation, 'sNaN', other)
        if modulo_is_nan == 2:
            return context._raise_error(InvalidOperation, 'sNaN', modulo)
        if self_is_nan:
            return self._fix_nan(context)
        if other_is_nan:
            return other._fix_nan(context)
        return modulo._fix_nan(context)
    if not (self._isinteger() and other._isinteger() and modulo._isinteger()):
        return context._raise_error(InvalidOperation, 'pow() 3rd argument not allowed unless all arguments are integers')
    if other < 0:
        return context._raise_error(InvalidOperation, 'pow() 2nd argument cannot be negative when 3rd argument specified')
    if not modulo:
        return context._raise_error(InvalidOperation, 'pow() 3rd argument cannot be 0')
    if modulo.adjusted() >= context.prec:
        return context._raise_error(InvalidOperation, 'insufficient precision: pow() 3rd argument must not have more than precision digits')
    if not other and (not self):
        return context._raise_error(InvalidOperation, 'at least one of pow() 1st argument and 2nd argument must be nonzero; 0**0 is not defined')
    if other._iseven():
        sign = 0
    else:
        sign = self._sign
    modulo = abs(int(modulo))
    base = _WorkRep(self.to_integral_value())
    exponent = _WorkRep(other.to_integral_value())
    base = base.int % modulo * pow(10, base.exp, modulo) % modulo
    for i in range(exponent.exp):
        base = pow(base, 10, modulo)
    base = pow(base, exponent.int, modulo)
    return _dec_from_triple(sign, str(base), 0)