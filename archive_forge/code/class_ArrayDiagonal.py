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
class ArrayDiagonal(_CodegenArrayAbstract):
    """
    Class to represent the diagonal operator.

    Explanation
    ===========

    In a 2-dimensional array it returns the diagonal, this looks like the
    operation:

    `A_{ij} \\rightarrow A_{ii}`

    The diagonal over axes 1 and 2 (the second and third) of the tensor product
    of two 2-dimensional arrays `A \\otimes B` is

    `\\Big[ A_{ab} B_{cd} \\Big]_{abcd} \\rightarrow \\Big[ A_{ai} B_{id} \\Big]_{adi}`

    In this last example the array expression has been reduced from
    4-dimensional to 3-dimensional. Notice that no contraction has occurred,
    rather there is a new index `i` for the diagonal, contraction would have
    reduced the array to 2 dimensions.

    Notice that the diagonalized out dimensions are added as new dimensions at
    the end of the indices.
    """

    def __new__(cls, expr, *diagonal_indices, **kwargs):
        expr = _sympify(expr)
        diagonal_indices = [Tuple(*sorted(i)) for i in diagonal_indices]
        canonicalize = kwargs.get('canonicalize', False)
        shape = get_shape(expr)
        if shape is not None:
            cls._validate(expr, *diagonal_indices, **kwargs)
            positions, shape = cls._get_positions_shape(shape, diagonal_indices)
        else:
            positions = None
        if len(diagonal_indices) == 0:
            return expr
        obj = Basic.__new__(cls, expr, *diagonal_indices)
        obj._positions = positions
        obj._subranks = _get_subranks(expr)
        obj._shape = shape
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        diagonal_indices = self.diagonal_indices
        trivial_diags = [i for i in diagonal_indices if len(i) == 1]
        if len(trivial_diags) > 0:
            trivial_pos = {e[0]: i for i, e in enumerate(diagonal_indices) if len(e) == 1}
            diag_pos = {e: i for i, e in enumerate(diagonal_indices) if len(e) > 1}
            diagonal_indices_short = [i for i in diagonal_indices if len(i) > 1]
            rank1 = get_rank(self)
            rank2 = len(diagonal_indices)
            rank3 = rank1 - rank2
            inv_permutation = []
            counter1 = 0
            indices_down = ArrayDiagonal._push_indices_down(diagonal_indices_short, list(range(rank1)), get_rank(expr))
            for i in indices_down:
                if i in trivial_pos:
                    inv_permutation.append(rank3 + trivial_pos[i])
                elif isinstance(i, (Integer, int)):
                    inv_permutation.append(counter1)
                    counter1 += 1
                else:
                    inv_permutation.append(rank3 + diag_pos[i])
            permutation = _af_invert(inv_permutation)
            if len(diagonal_indices_short) > 0:
                return _permute_dims(_array_diagonal(expr, *diagonal_indices_short), permutation)
            else:
                return _permute_dims(expr, permutation)
        if isinstance(expr, ArrayAdd):
            return self._ArrayDiagonal_denest_ArrayAdd(expr, *diagonal_indices)
        if isinstance(expr, ArrayDiagonal):
            return self._ArrayDiagonal_denest_ArrayDiagonal(expr, *diagonal_indices)
        if isinstance(expr, PermuteDims):
            return self._ArrayDiagonal_denest_PermuteDims(expr, *diagonal_indices)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            positions, shape = self._get_positions_shape(expr.shape, diagonal_indices)
            return ZeroArray(*shape)
        return self.func(expr, *diagonal_indices, canonicalize=False)

    @staticmethod
    def _validate(expr, *diagonal_indices, **kwargs):
        shape = get_shape(expr)
        for i in diagonal_indices:
            if any((j >= len(shape) for j in i)):
                raise ValueError('index is larger than expression shape')
            if len({shape[j] for j in i}) != 1:
                raise ValueError('diagonalizing indices of different dimensions')
            if not kwargs.get('allow_trivial_diags', False) and len(i) <= 1:
                raise ValueError('need at least two axes to diagonalize')
            if len(set(i)) != len(i):
                raise ValueError('axis index cannot be repeated')

    @staticmethod
    def _remove_trivial_dimensions(shape, *diagonal_indices):
        return [tuple((j for j in i)) for i in diagonal_indices if shape[i[0]] != 1]

    @property
    def expr(self):
        return self.args[0]

    @property
    def diagonal_indices(self):
        return self.args[1:]

    @staticmethod
    def _flatten(expr, *outer_diagonal_indices):
        inner_diagonal_indices = expr.diagonal_indices
        all_inner = [j for i in inner_diagonal_indices for j in i]
        all_inner.sort()
        total_rank = _get_subrank(expr)
        inner_rank = len(all_inner)
        outer_rank = total_rank - inner_rank
        shifts = [0 for i in range(outer_rank)]
        counter = 0
        pointer = 0
        for i in range(outer_rank):
            while pointer < inner_rank and counter >= all_inner[pointer]:
                counter += 1
                pointer += 1
            shifts[i] += pointer
            counter += 1
        outer_diagonal_indices = tuple((tuple((shifts[j] + j for j in i)) for i in outer_diagonal_indices))
        diagonal_indices = inner_diagonal_indices + outer_diagonal_indices
        return _array_diagonal(expr.expr, *diagonal_indices)

    @classmethod
    def _ArrayDiagonal_denest_ArrayAdd(cls, expr, *diagonal_indices):
        return _array_add(*[_array_diagonal(arg, *diagonal_indices) for arg in expr.args])

    @classmethod
    def _ArrayDiagonal_denest_ArrayDiagonal(cls, expr, *diagonal_indices):
        return cls._flatten(expr, *diagonal_indices)

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

    def _push_indices_down_nonstatic(self, indices):
        transform = lambda x: self._positions[x] if x < len(self._positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    def _push_indices_up_nonstatic(self, indices):

        def transform(x):
            for i, e in enumerate(self._positions):
                if isinstance(e, int) and x == e or (isinstance(e, tuple) and x in e):
                    return i
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_down(cls, diagonal_indices, indices, rank):
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)
        transform = lambda x: positions[x] if x < len(positions) else None
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _push_indices_up(cls, diagonal_indices, indices, rank):
        positions, shape = cls._get_positions_shape(range(rank), diagonal_indices)

        def transform(x):
            for i, e in enumerate(positions):
                if isinstance(e, int) and x == e or (isinstance(e, (tuple, Tuple)) and x in e):
                    return i
        return _apply_recursively_over_nested_lists(transform, indices)

    @classmethod
    def _get_positions_shape(cls, shape, diagonal_indices):
        data1 = tuple(((i, shp) for i, shp in enumerate(shape) if not any((i in j for j in diagonal_indices))))
        pos1, shp1 = zip(*data1) if data1 else ((), ())
        data2 = tuple(((i, shape[i[0]]) for i in diagonal_indices))
        pos2, shp2 = zip(*data2) if data2 else ((), ())
        positions = pos1 + pos2
        shape = shp1 + shp2
        return (positions, shape)

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, 'as_explicit'):
            expr = expr.as_explicit()
        return tensordiagonal(expr, *self.diagonal_indices)