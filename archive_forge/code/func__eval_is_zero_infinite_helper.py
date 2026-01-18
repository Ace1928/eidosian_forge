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
def _eval_is_zero_infinite_helper(self):
    seen_zero = seen_infinite = False
    for a in self.args:
        if a.is_zero:
            if seen_infinite is not False:
                return (None, None)
            seen_zero = True
        elif a.is_infinite:
            if seen_zero is not False:
                return (None, None)
            seen_infinite = True
        else:
            if seen_zero is False and a.is_zero is None:
                if seen_infinite is not False:
                    return (None, None)
                seen_zero = None
            if seen_infinite is False and a.is_infinite is None:
                if seen_zero is not False:
                    return (None, None)
                seen_infinite = None
    return (seen_zero, seen_infinite)