import itertools
from collections.abc import Iterable
from sympy.core._print_helpers import Printable
from sympy.core.containers import Tuple
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.array.dense_ndim_array import DenseNDimArray, ImmutableDenseNDimArray
from sympy.tensor.array.sparse_ndim_array import SparseNDimArray
def _util_contraction_diagonal(array, *contraction_or_diagonal_axes):
    array = _arrayfy(array)
    taken_dims = set()
    for axes_group in contraction_or_diagonal_axes:
        if not isinstance(axes_group, Iterable):
            raise ValueError('collections of contraction/diagonal axes expected')
        dim = array.shape[axes_group[0]]
        for d in axes_group:
            if d in taken_dims:
                raise ValueError('dimension specified more than once')
            if dim != array.shape[d]:
                raise ValueError('cannot contract or diagonalize between axes of different dimension')
            taken_dims.add(d)
    rank = array.rank()
    remaining_shape = [dim for i, dim in enumerate(array.shape) if i not in taken_dims]
    cum_shape = [0] * rank
    _cumul = 1
    for i in range(rank):
        cum_shape[rank - i - 1] = _cumul
        _cumul *= int(array.shape[rank - i - 1])
    remaining_indices = [[cum_shape[i] * j for j in range(array.shape[i])] for i in range(rank) if i not in taken_dims]
    summed_deltas = []
    for axes_group in contraction_or_diagonal_axes:
        lidx = []
        for js in range(array.shape[axes_group[0]]):
            lidx.append(sum([cum_shape[ig] * js for ig in axes_group]))
        summed_deltas.append(lidx)
    return (array, remaining_indices, remaining_shape, summed_deltas)