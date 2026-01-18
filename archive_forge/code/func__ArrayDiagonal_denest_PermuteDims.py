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
def _ArrayDiagonal_denest_PermuteDims(cls, expr: PermuteDims, *diagonal_indices):
    back_diagonal_indices = [[expr.permutation(j) for j in i] for i in diagonal_indices]
    nondiag = [i for i in range(get_rank(expr)) if not any((i in j for j in diagonal_indices))]
    back_nondiag = [expr.permutation(i) for i in nondiag]
    remap = {e: i for i, e in enumerate(sorted(back_nondiag))}
    new_permutation1 = [remap[i] for i in back_nondiag]
    shift = len(new_permutation1)
    diag_block_perm = [i + shift for i in range(len(back_diagonal_indices))]
    new_permutation = new_permutation1 + diag_block_perm
    return _permute_dims(_array_diagonal(expr.expr, *back_diagonal_indices), new_permutation)