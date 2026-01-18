import itertools
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Not
from sympy.core.parameters import global_parameters
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.stats import variance, covariance
from sympy.stats.rv import (RandomSymbol, pspace, dependent,
@classmethod
def _expand_single_argument(cls, expr):
    if isinstance(expr, RandomSymbol):
        return [(S.One, expr)]
    elif isinstance(expr, Add):
        outval = []
        for a in expr.args:
            if isinstance(a, Mul):
                outval.append(cls._get_mul_nonrv_rv_tuple(a))
            elif is_random(a):
                outval.append((S.One, a))
        return outval
    elif isinstance(expr, Mul):
        return [cls._get_mul_nonrv_rv_tuple(expr)]
    elif is_random(expr):
        return [(S.One, expr)]