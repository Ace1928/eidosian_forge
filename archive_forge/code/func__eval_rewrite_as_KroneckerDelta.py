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
def _eval_rewrite_as_KroneckerDelta(self, *args):
    from sympy.functions.special.tensor_functions import KroneckerDelta
    rules = {And: [False, False], Or: [True, True], Not: [True, False], Eq: [None, None], Ne: [None, None]}

    class UnrecognizedCondition(Exception):
        pass

    def rewrite(cond):
        if isinstance(cond, Eq):
            return KroneckerDelta(*cond.args)
        if isinstance(cond, Ne):
            return 1 - KroneckerDelta(*cond.args)
        cls, args = (type(cond), cond.args)
        if cls not in rules:
            raise UnrecognizedCondition(cls)
        b1, b2 = rules[cls]
        k = Mul(*[1 - rewrite(c) for c in args]) if b1 else Mul(*[rewrite(c) for c in args])
        if b2:
            return 1 - k
        return k
    conditions = []
    true_value = None
    for value, cond in args:
        if type(cond) in rules:
            conditions.append((value, cond))
        elif cond is S.true:
            if true_value is None:
                true_value = value
        else:
            return
    if true_value is not None:
        result = true_value
        for value, cond in conditions[::-1]:
            try:
                k = rewrite(cond)
                result = k * value + (1 - k) * result
            except UnrecognizedCondition:
                return
        return result