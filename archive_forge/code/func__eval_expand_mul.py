from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from itertools import product
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
def _eval_expand_mul(self, **hints):
    from sympy.simplify.radsimp import fraction
    expr = self
    n, d = fraction(expr)
    if d.is_Mul:
        n, d = [i._eval_expand_mul(**hints) if i.is_Mul else i for i in (n, d)]
    expr = n / d
    if not expr.is_Mul:
        return expr
    plain, sums, rewrite = ([], [], False)
    for factor in expr.args:
        if factor.is_Add:
            sums.append(factor)
            rewrite = True
        elif factor.is_commutative:
            plain.append(factor)
        else:
            sums.append(Basic(factor))
    if not rewrite:
        return expr
    else:
        plain = self.func(*plain)
        if sums:
            deep = hints.get('deep', False)
            terms = self.func._expandsums(sums)
            args = []
            for term in terms:
                t = self.func(plain, term)
                if t.is_Mul and any((a.is_Add for a in t.args)) and deep:
                    t = t._eval_expand_mul()
                args.append(t)
            return Add(*args)
        else:
            return plain