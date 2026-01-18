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
def _sort_fully_contracted_args(cls, expr, contraction_indices):
    if expr.shape is None:
        return (expr, contraction_indices)
    cumul = list(accumulate([0] + expr.subranks))
    index_blocks = [list(range(cumul[i], cumul[i + 1])) for i in range(len(expr.args))]
    contraction_indices_flat = {j for i in contraction_indices for j in i}
    fully_contracted = [all((j in contraction_indices_flat for j in range(cumul[i], cumul[i + 1]))) for i, arg in enumerate(expr.args)]
    new_pos = sorted(range(len(expr.args)), key=lambda x: (0, default_sort_key(expr.args[x])) if fully_contracted[x] else (1,))
    new_args = [expr.args[i] for i in new_pos]
    new_index_blocks_flat = [j for i in new_pos for j in index_blocks[i]]
    index_permutation_array_form = _af_invert(new_index_blocks_flat)
    new_contraction_indices = [tuple((index_permutation_array_form[j] for j in i)) for i in contraction_indices]
    new_contraction_indices = _sort_contraction_indices(new_contraction_indices)
    return (_array_tensor_product(*new_args), new_contraction_indices)