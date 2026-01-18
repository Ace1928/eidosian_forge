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
@staticmethod
def _tensAdd_check(args):

    def get_indices_set(x: Expr) -> set[TensorIndex]:
        if isinstance(x, TensExpr):
            return set(x.get_free_indices())
        return set()
    indices0 = get_indices_set(args[0])
    list_indices = [get_indices_set(arg) for arg in args[1:]]
    if not all((x == indices0 for x in list_indices)):
        raise ValueError('all tensors must have the same indices')