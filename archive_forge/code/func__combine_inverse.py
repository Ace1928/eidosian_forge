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
@staticmethod
def _combine_inverse(lhs, rhs):
    """
        Returns lhs - rhs, but treats oo like a symbol so oo - oo
        returns 0, instead of a nan.
        """
    from sympy.simplify.simplify import signsimp
    inf = (S.Infinity, S.NegativeInfinity)
    if lhs.has(*inf) or rhs.has(*inf):
        from .symbol import Dummy
        oo = Dummy('oo')
        reps = {S.Infinity: oo, S.NegativeInfinity: -oo}
        ireps = {v: k for k, v in reps.items()}
        eq = lhs.xreplace(reps) - rhs.xreplace(reps)
        if eq.has(oo):
            eq = eq.replace(lambda x: x.is_Pow and x.base is oo, lambda x: x.base)
        rv = eq.xreplace(ireps)
    else:
        rv = lhs - rhs
    srv = signsimp(rv)
    return srv if srv.is_Number else rv