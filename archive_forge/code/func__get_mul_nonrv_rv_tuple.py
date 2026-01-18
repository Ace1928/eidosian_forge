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
def _get_mul_nonrv_rv_tuple(cls, m):
    rv = []
    nonrv = []
    for a in m.args:
        if is_random(a):
            rv.append(a)
        else:
            nonrv.append(a)
    return (Mul.fromiter(nonrv), Mul.fromiter(rv))