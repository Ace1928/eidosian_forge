from collections import defaultdict
from sympy.core import Add, S
from sympy.core.function import diff, expand, _mexpand, expand_mul
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Wild
from sympy.functions import exp, cos, cosh, im, log, re, sin, sinh, \
from sympy.integrals import Integral
from sympy.polys import (Poly, RootOf, rootof, roots)
from sympy.simplify import collect, simplify, separatevars, powsimp, trigsimp # type: ignore
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.matrices import wronskian
from .subscheck import sub_func_doit
from sympy.solvers.ode.ode import get_numbered_constants
def _get_trial_set(expr, x, exprs=set()):
    """
        Returns a set of trial terms for undetermined coefficients.

        The idea behind undetermined coefficients is that the terms expression
        repeat themselves after a finite number of derivatives, except for the
        coefficients (they are linearly dependent).  So if we collect these,
        we should have the terms of our trial function.
        """

    def _remove_coefficient(expr, x):
        """
            Returns the expression without a coefficient.

            Similar to expr.as_independent(x)[1], except it only works
            multiplicatively.
            """
        term = S.One
        if expr.is_Mul:
            for i in expr.args:
                if i.has(x):
                    term *= i
        elif expr.has(x):
            term = expr
        return term
    expr = expand_mul(expr)
    if expr.is_Add:
        for term in expr.args:
            if _remove_coefficient(term, x) in exprs:
                pass
            else:
                exprs.add(_remove_coefficient(term, x))
                exprs = exprs.union(_get_trial_set(term, x, exprs))
    else:
        term = _remove_coefficient(expr, x)
        tmpset = exprs.union({term})
        oldset = set()
        while tmpset != oldset:
            oldset = tmpset.copy()
            expr = expr.diff(x)
            term = _remove_coefficient(expr, x)
            if term.is_Add:
                tmpset = tmpset.union(_get_trial_set(term, x, tmpset))
            else:
                tmpset.add(term)
        exprs = tmpset
    return exprs