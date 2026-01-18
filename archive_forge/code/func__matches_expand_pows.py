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
@staticmethod
def _matches_expand_pows(arg_list):
    new_args = []
    for arg in arg_list:
        if arg.is_Pow and arg.exp > 0:
            new_args.extend([arg.base] * arg.exp)
        else:
            new_args.append(arg)
    return new_args