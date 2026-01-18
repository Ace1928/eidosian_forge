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
def _solve_relational(r):
    if sym not in r.free_symbols:
        return nonsymfail(r)
    try:
        rv = _solve_inequality(r, sym)
    except NotImplementedError:
        return (False, 'Unable to solve relational %s for %s.' % (r, sym))
    if isinstance(rv, Relational):
        free = rv.args[1].free_symbols
        if rv.args[0] != sym or sym in free:
            return (False, 'Unable to solve relational %s for %s.' % (r, sym))
        if rv.rel_op == '==':
            rv = S.false
        elif rv.rel_op == '!=':
            try:
                rv = Or(sym < rv.rhs, sym > rv.rhs)
            except TypeError:
                rv = S.true
    elif rv == (S.NegativeInfinity < sym) & (sym < S.Infinity):
        rv = S.true
    return (True, rv)