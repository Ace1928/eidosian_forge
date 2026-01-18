from __future__ import annotations
from functools import reduce
from sympy.core import S, sympify, Dummy, Mod
from sympy.core.cache import cacheit
from sympy.core.function import Function, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
from sympy.core.numbers import Integer, pi, I
from sympy.core.relational import Eq
from sympy.external.gmpy import HAS_GMPY, gmpy
from sympy.ntheory import sieve
from sympy.polys.polytools import Poly
from math import factorial as _factorial, prod, sqrt as _sqrt
class CombinatorialFunction(Function):
    """Base class for combinatorial functions. """

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.combsimp import combsimp
        expr = combsimp(self)
        measure = kwargs['measure']
        if measure(expr) <= kwargs['ratio'] * measure(self):
            return expr
        return self