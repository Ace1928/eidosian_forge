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
@classmethod
def _ArrayContraction_denest_ArrayDiagonal(cls, expr: 'ArrayDiagonal', *contraction_indices):
    diagonal_indices = list(expr.diagonal_indices)
    down_contraction_indices = expr._push_indices_down(expr.diagonal_indices, contraction_indices, get_rank(expr.expr))
    down_contraction_indices = [[k for j in i for k in (j if isinstance(j, (tuple, Tuple)) else [j])] for i in down_contraction_indices]
    new_contraction_indices = []
    for contr_indgrp in down_contraction_indices:
        ind = contr_indgrp[:]
        for j, diag_indgrp in enumerate(diagonal_indices):
            if diag_indgrp is None:
                continue
            if any((i in diag_indgrp for i in contr_indgrp)):
                ind.extend(diag_indgrp)
                diagonal_indices[j] = None
        new_contraction_indices.append(sorted(set(ind)))
    new_diagonal_indices_down = [i for i in diagonal_indices if i is not None]
    new_diagonal_indices = ArrayContraction._push_indices_up(new_contraction_indices, new_diagonal_indices_down)
    return _array_diagonal(_array_contraction(expr.expr, *new_contraction_indices), *new_diagonal_indices)