import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
class TrueDiv(sympy.Function):

    @classmethod
    def eval(cls, base, divisor):
        if divisor.is_zero:
            raise ZeroDivisionError('division by zero')
        else:
            return base / divisor