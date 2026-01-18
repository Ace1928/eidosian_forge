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
def _lower_contraction_to_addends(cls, expr, contraction_indices):
    if isinstance(expr, ArrayAdd):
        raise NotImplementedError()
    if not isinstance(expr, ArrayTensorProduct):
        return (expr, contraction_indices)
    subranks = expr.subranks
    cumranks = list(accumulate([0] + subranks))
    contraction_indices_remaining = []
    contraction_indices_args = [[] for i in expr.args]
    backshift = set()
    for i, contraction_group in enumerate(contraction_indices):
        for j in range(len(expr.args)):
            if not isinstance(expr.args[j], ArrayAdd):
                continue
            if all((cumranks[j] <= k < cumranks[j + 1] for k in contraction_group)):
                contraction_indices_args[j].append([k - cumranks[j] for k in contraction_group])
                backshift.update(contraction_group)
                break
        else:
            contraction_indices_remaining.append(contraction_group)
    if len(contraction_indices_remaining) == len(contraction_indices):
        return (expr, contraction_indices)
    total_rank = get_rank(expr)
    shifts = list(accumulate([1 if i in backshift else 0 for i in range(total_rank)]))
    contraction_indices_remaining = [Tuple.fromiter((j - shifts[j] for j in i)) for i in contraction_indices_remaining]
    ret = _array_tensor_product(*[_array_contraction(arg, *contr) for arg, contr in zip(expr.args, contraction_indices_args)])
    return (ret, contraction_indices_remaining)