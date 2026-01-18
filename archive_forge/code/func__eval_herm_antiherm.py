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
def _eval_herm_antiherm(self, herm):
    for t in self.args:
        if t.is_hermitian is None or t.is_antihermitian is None:
            return
        if t.is_hermitian:
            continue
        elif t.is_antihermitian:
            herm = not herm
        else:
            return
    if herm is not False:
        return herm
    is_zero = self._eval_is_zero()
    if is_zero:
        return True
    elif is_zero is False:
        return herm