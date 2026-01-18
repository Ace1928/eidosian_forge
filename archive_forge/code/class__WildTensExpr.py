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
class _WildTensExpr(Basic):
    """
    INTERNAL USE ONLY

    This is an object that helps with replacement of WildTensors in expressions.
    When this object is set as the tensor_head of a WildTensor, it replaces the
    WildTensor by a TensExpr (passed when initializing this object).

    Examples
    ========
    >>> from sympy.tensor.tensor import WildTensorHead, TensorIndex, TensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)
    >>> K = TensorHead('K', [R3])
    >>> print( ( K(p) ).replace( W(p), W(q)*W(-q)*W(p) ) )
    K(R_0)*K(-R_0)*K(p)

    """

    def __init__(self, expr):
        if not isinstance(expr, TensExpr):
            raise TypeError('_WildTensExpr expects a TensExpr as argument')
        self.expr = expr

    def __call__(self, *indices):
        return self.expr._replace_indices(dict(zip(self.expr.get_free_indices(), indices)))

    def __neg__(self):
        return self.func(self.expr * S.NegativeOne)

    def __abs__(self):
        raise NotImplementedError

    def __add__(self, other):
        if other.func != self.func:
            raise TypeError(f'Cannot add {self.func} to {other.func}')
        return self.func(self.expr + other.expr)

    def __radd__(self, other):
        if other.func != self.func:
            raise TypeError(f'Cannot add {self.func} to {other.func}')
        return self.func(other.expr + self.expr)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __rpow__(self, other):
        raise NotImplementedError