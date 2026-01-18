import itertools
from sympy.calculus.util import (continuous_domain, periodicity,
from sympy.core import Symbol, Dummy, sympify
from sympy.core.exprtools import factor_terms
from sympy.core.relational import Relational, Eq, Ge, Lt
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.core.singleton import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.complexes import im, Abs
from sympy.logic import And
from sympy.polys import Poly, PolynomialError, parallel_poly_from_expr
from sympy.polys.polyutils import _nsort
from sympy.solvers.solveset import solvify, solveset
from sympy.utilities.iterables import sift, iterable
from sympy.utilities.misc import filldedent
def _bottom_up_scan(expr):
    exprs = []
    if expr.is_Add or expr.is_Mul:
        op = expr.func
        for arg in expr.args:
            _exprs = _bottom_up_scan(arg)
            if not exprs:
                exprs = _exprs
            else:
                exprs = [(op(expr, _expr), conds + _conds) for (expr, conds), (_expr, _conds) in itertools.product(exprs, _exprs)]
    elif expr.is_Pow:
        n = expr.exp
        if not n.is_Integer:
            raise ValueError('Only Integer Powers are allowed on Abs.')
        exprs.extend(((expr ** n, conds) for expr, conds in _bottom_up_scan(expr.base)))
    elif isinstance(expr, Abs):
        _exprs = _bottom_up_scan(expr.args[0])
        for expr, conds in _exprs:
            exprs.append((expr, conds + [Ge(expr, 0)]))
            exprs.append((-expr, conds + [Lt(expr, 0)]))
    else:
        exprs = [(expr, [])]
    return exprs