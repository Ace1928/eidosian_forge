from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from operator import attrgetter
from .basic import Basic
from .parameters import global_parameters
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .numbers import ilcm, igcd, equal_valued
from .expr import Expr
from .kind import UndefinedKind
from sympy.utilities.iterables import is_sequence, sift
from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational
def _eval_is_extended_negative(self):
    if self.is_number:
        return super()._eval_is_extended_negative()
    c, a = self.as_coeff_Add()
    if not c.is_zero:
        from .exprtools import _monotonic_sign
        v = _monotonic_sign(a)
        if v is not None:
            s = v + c
            if s != self and s.is_extended_negative and a.is_extended_nonpositive:
                return True
            if len(self.free_symbols) == 1:
                v = _monotonic_sign(self)
                if v is not None and v != self and v.is_extended_negative:
                    return True
    neg = nonpos = nonneg = unknown_sign = False
    saw_INF = set()
    args = [a for a in self.args if not a.is_zero]
    if not args:
        return False
    for a in args:
        isneg = a.is_extended_negative
        infinite = a.is_infinite
        if infinite:
            saw_INF.add(fuzzy_or((isneg, a.is_extended_nonpositive)))
            if True in saw_INF and False in saw_INF:
                return
        if isneg:
            neg = True
            continue
        elif a.is_extended_nonpositive:
            nonpos = True
            continue
        elif a.is_extended_nonnegative:
            nonneg = True
            continue
        if infinite is None:
            return
        unknown_sign = True
    if saw_INF:
        if len(saw_INF) > 1:
            return
        return saw_INF.pop()
    elif unknown_sign:
        return
    elif not nonneg and (not nonpos) and neg:
        return True
    elif not nonneg and neg:
        return True
    elif not neg and (not nonpos):
        return False