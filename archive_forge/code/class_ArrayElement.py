import collections.abc
import operator
from collections import defaultdict, Counter
from functools import reduce
import itertools
from itertools import accumulate
from typing import Optional, List, Tuple as tTuple
import typing
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import (Dummy, Symbol)
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.diagonal import diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, \
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.core.sympify import _sympify
class ArrayElement(Expr):
    """
    An element of an array.
    """
    _diff_wrt = True
    is_symbol = True
    is_commutative = True

    def __new__(cls, name, indices):
        if isinstance(name, str):
            name = Symbol(name)
        name = _sympify(name)
        if not isinstance(indices, collections.abc.Iterable):
            indices = (indices,)
        indices = _sympify(tuple(indices))
        cls._check_shape(name, indices)
        obj = Expr.__new__(cls, name, indices)
        return obj

    @classmethod
    def _check_shape(cls, name, indices):
        indices = tuple(indices)
        if hasattr(name, 'shape'):
            index_error = IndexError('number of indices does not match shape of the array')
            if len(indices) != len(name.shape):
                raise index_error
            if any(((i >= s) == True for i, s in zip(indices, name.shape))):
                raise ValueError('shape is out of bounds')
        if any(((i < 0) == True for i in indices)):
            raise ValueError('shape contains negative values')

    @property
    def name(self):
        return self._args[0]

    @property
    def indices(self):
        return self._args[1]

    def _eval_derivative(self, s):
        if not isinstance(s, ArrayElement):
            return S.Zero
        if s == self:
            return S.One
        if s.name != self.name:
            return S.Zero
        return Mul.fromiter((KroneckerDelta(i, j) for i, j in zip(self.indices, s.indices)))