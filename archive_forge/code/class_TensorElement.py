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
class TensorElement(TensExpr):
    """
    Tensor with evaluated components.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> i, j, k = symbols("i j k")
    >>> A = TensorHead("A", [L, L], TensorSymmetry.fully_symmetric(2))
    >>> A(i, j).get_free_indices()
    [i, j]

    If we want to set component ``i`` to a specific value, use the
    ``TensorElement`` class:

    >>> from sympy.tensor.tensor import TensorElement
    >>> te = TensorElement(A(i, j), {i: 2})

    As index ``i`` has been accessed (``{i: 2}`` is the evaluation of its 3rd
    element), the free indices will only contain ``j``:

    >>> te.get_free_indices()
    [j]
    """

    def __new__(cls, expr, index_map):
        if not isinstance(expr, Tensor):
            if not isinstance(expr, TensExpr):
                raise TypeError('%s is not a tensor expression' % expr)
            return expr.func(*[TensorElement(arg, index_map) for arg in expr.args])
        expr_free_indices = expr.get_free_indices()
        name_translation = {i.args[0]: i for i in expr_free_indices}
        index_map = {name_translation.get(index, index): value for index, value in index_map.items()}
        index_map = {index: value for index, value in index_map.items() if index in expr_free_indices}
        if len(index_map) == 0:
            return expr
        free_indices = [i for i in expr_free_indices if i not in index_map.keys()]
        index_map = Dict(index_map)
        obj = TensExpr.__new__(cls, expr, index_map)
        obj._free_indices = free_indices
        return obj

    @property
    def free(self):
        return [(index, i) for i, index in enumerate(self.get_free_indices())]

    @property
    def dum(self):
        return []

    @property
    def expr(self):
        return self._args[0]

    @property
    def index_map(self):
        return self._args[1]

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    def get_free_indices(self):
        return self._free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        return self.xreplace(repl)

    def get_indices(self):
        return self.get_free_indices()

    def _extract_data(self, replacement_dict):
        ret_indices, array = self.expr._extract_data(replacement_dict)
        index_map = self.index_map
        slice_tuple = tuple((index_map.get(i, slice(None)) for i in ret_indices))
        ret_indices = [i for i in ret_indices if i not in index_map]
        array = array.__getitem__(slice_tuple)
        return (ret_indices, array)