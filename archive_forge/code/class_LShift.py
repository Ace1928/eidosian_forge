import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
class LShift(sympy.Function):

    @classmethod
    def eval(cls, base, shift):
        if shift < 0:
            raise ValueError('negative shift count')
        return base * 2 ** shift