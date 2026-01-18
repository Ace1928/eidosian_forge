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
@classmethod
def __eval_cond(cls, cond):
    """Return the truth value of the condition."""
    if cond == True:
        return True
    if isinstance(cond, Eq):
        try:
            diff = cond.lhs - cond.rhs
            if diff.is_commutative:
                return diff.is_zero
        except TypeError:
            pass