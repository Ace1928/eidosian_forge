from __future__ import annotations
from typing import Callable
from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
from sympy.core.logic import fuzzy_not
from sympy.logic.boolalg import Boolean
from sympy.assumptions import ask, Q  # type: ignore
def _refine_reim(expr, assumptions):
    expanded = expr.expand(complex=True)
    if expanded != expr:
        refined = refine(expanded, assumptions)
        if refined != expanded:
            return refined
    return None