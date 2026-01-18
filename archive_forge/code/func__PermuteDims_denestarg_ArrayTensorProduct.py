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
def _PermuteDims_denestarg_ArrayTensorProduct(cls, expr, permutation):
    perm_image_form = _af_invert(permutation.array_form)
    args = list(expr.args)
    cumul = list(accumulate([0] + expr.subranks))
    perm_image_form_in_components = [perm_image_form[cumul[i]:cumul[i + 1]] for i in range(len(args))]
    ps = [(i, sorted(comp)) for i, comp in enumerate(perm_image_form_in_components)]
    ps.sort(key=lambda x: x[1])
    perm_args_image_form = [i[0] for i in ps]
    args_sorted = [args[i] for i in perm_args_image_form]
    perm_image_form_sorted_args = [perm_image_form_in_components[i] for i in perm_args_image_form]
    new_permutation = Permutation(_af_invert([j for i in perm_image_form_sorted_args for j in i]))
    return (_array_tensor_product(*args_sorted), new_permutation)