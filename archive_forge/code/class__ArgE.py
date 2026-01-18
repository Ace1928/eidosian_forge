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
class _ArgE:
    """
    The ``_ArgE`` object contains references to the array expression
    (``.element``) and a list containing the information about index
    contractions (``.indices``).

    Index contractions are numbered and contracted indices show the number of
    the contraction. Uncontracted indices have ``None`` value.

    For example:
    ``_ArgE(M, [None, 3])``
    This object means that expression ``M`` is part of an array contraction
    and has two indices, the first is not contracted (value ``None``),
    the second index is contracted to the 4th (i.e. number ``3``) group of the
    array contraction object.
    """
    indices: List[Optional[int]]

    def __init__(self, element, indices: Optional[List[Optional[int]]]=None):
        self.element = element
        if indices is None:
            self.indices = [None for i in range(get_rank(element))]
        else:
            self.indices = indices

    def __str__(self):
        return '_ArgE(%s, %s)' % (self.element, self.indices)
    __repr__ = __str__