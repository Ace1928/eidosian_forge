from sympy.assumptions import Q, ask
from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import (ImaginaryUnit, Infinity, Integer, NaN,
from sympy.functions import Abs, im, re
from sympy.ntheory import isprime
from sympy.multipledispatch import MDNotImplementedError
from ..predicates.ntheory import (PrimePredicate, CompositePredicate,
def _EvenPredicate_number(expr, assumptions):
    try:
        i = int(expr.round())
        if not (expr - i).equals(0):
            raise TypeError
    except TypeError:
        return False
    if isinstance(expr, (float, Float)):
        return False
    return i % 2 == 0