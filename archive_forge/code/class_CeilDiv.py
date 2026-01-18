import sympy
from sympy import S
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
class CeilDiv(sympy.Function):
    """
    Div used in indexing that rounds up.
    """
    is_integer = True

    def __new__(cls, base, divisor):
        if sympy.gcd(base, divisor) == divisor:
            return CleanDiv(base, divisor)
        else:
            return FloorDiv(base + (divisor - 1), divisor)