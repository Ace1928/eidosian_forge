from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
def _get_indices_to_args_pos(self):
    """
        Get a dict mapping the index position to TensMul's argument number.
        """
    pos_map = {}
    pos_counter = 0
    for arg_i, arg in enumerate(self.args):
        if not isinstance(arg, TensExpr):
            continue
        assert isinstance(arg, Tensor)
        for i in range(arg.ext_rank):
            pos_map[pos_counter] = arg_i
            pos_counter += 1
    return pos_map