from typing import Tuple as tTuple
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
def _unpolarify(eq, exponents_only, pause=False):
    if not isinstance(eq, Basic) or eq.is_Atom:
        return eq
    if not pause:
        from sympy.functions.elementary.exponential import exp, exp_polar
        if isinstance(eq, exp_polar):
            return exp(_unpolarify(eq.exp, exponents_only))
        if isinstance(eq, principal_branch) and eq.args[1] == 2 * pi:
            return _unpolarify(eq.args[0], exponents_only)
        if eq.is_Add or eq.is_Mul or eq.is_Boolean or (eq.is_Relational and (eq.rel_op in ('==', '!=') and 0 in eq.args or eq.rel_op not in ('==', '!='))):
            return eq.func(*[_unpolarify(x, exponents_only) for x in eq.args])
        if isinstance(eq, polar_lift):
            return _unpolarify(eq.args[0], exponents_only)
    if eq.is_Pow:
        expo = _unpolarify(eq.exp, exponents_only)
        base = _unpolarify(eq.base, exponents_only, not (expo.is_integer and (not pause)))
        return base ** expo
    if eq.is_Function and getattr(eq.func, 'unbranched', False):
        return eq.func(*[_unpolarify(x, exponents_only, exponents_only) for x in eq.args])
    return eq.func(*[_unpolarify(x, exponents_only, True) for x in eq.args])