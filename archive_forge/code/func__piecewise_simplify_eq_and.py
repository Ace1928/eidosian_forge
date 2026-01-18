from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def _piecewise_simplify_eq_and(args):
    """
    Try to simplify conditions and the expression for
    equalities that are part of the condition, e.g.
    Piecewise((n, And(Eq(n,0), Eq(n + m, 0))), (1, True))
    -> Piecewise((0, And(Eq(n, 0), Eq(m, 0))), (1, True))
    """
    for i, (expr, cond) in enumerate(args):
        if isinstance(cond, And):
            eqs, other = sift(cond.args, lambda i: isinstance(i, Eq), binary=True)
        elif isinstance(cond, Eq):
            eqs, other = ([cond], [])
        else:
            eqs = other = []
        if eqs:
            eqs = list(ordered(eqs))
            for j, e in enumerate(eqs):
                if _blessed(e):
                    expr = expr.subs(*e.args)
                    eqs[j + 1:] = [ei.subs(*e.args) for ei in eqs[j + 1:]]
                    other = [ei.subs(*e.args) for ei in other]
            cond = And(*eqs + other)
            args[i] = args[i].func(expr, cond)
    return args