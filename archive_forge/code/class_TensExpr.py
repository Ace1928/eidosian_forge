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
class TensExpr(Expr, ABC):
    """
    Abstract base class for tensor expressions

    Notes
    =====

    A tensor expression is an expression formed by tensors;
    currently the sums of tensors are distributed.

    A ``TensExpr`` can be a ``TensAdd`` or a ``TensMul``.

    ``TensMul`` objects are formed by products of component tensors,
    and include a coefficient, which is a SymPy expression.


    In the internal representation contracted indices are represented
    by ``(ipos1, ipos2, icomp1, icomp2)``, where ``icomp1`` is the position
    of the component tensor with contravariant index, ``ipos1`` is the
    slot which the index occupies in that component tensor.

    Contracted indices are therefore nameless in the internal representation.
    """
    _op_priority = 12.0
    is_commutative = False

    def __neg__(self):
        return self * S.NegativeOne

    def __abs__(self):
        raise NotImplementedError

    def __add__(self, other):
        return TensAdd(self, other).doit()

    def __radd__(self, other):
        return TensAdd(other, self).doit()

    def __sub__(self, other):
        return TensAdd(self, -other).doit()

    def __rsub__(self, other):
        return TensAdd(other, -self).doit()

    def __mul__(self, other):
        """
        Multiply two tensors using Einstein summation convention.

        Explanation
        ===========

        If the two tensors have an index in common, one contravariant
        and the other covariant, in their product the indices are summed

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t1 = p(m0)
        >>> t2 = q(-m0)
        >>> t1*t2
        p(L_0)*q(-L_0)
        """
        return TensMul(self, other).doit()

    def __rmul__(self, other):
        return TensMul(other, self).doit()

    def __truediv__(self, other):
        other = _sympify(other)
        if isinstance(other, TensExpr):
            raise ValueError('cannot divide by a tensor')
        return TensMul(self, S.One / other).doit()

    def __rtruediv__(self, other):
        raise ValueError('cannot divide by a tensor')

    def __pow__(self, other):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError('No power without ndarray data.')
            from .array import tensorproduct, tensorcontraction
            free = self.free
            marray = self.data
            mdim = marray.rank()
            for metric in free:
                marray = tensorcontraction(tensorproduct(marray, metric[0].tensor_index_type.data, marray), (0, mdim), (mdim + 1, mdim + 2))
            return marray ** (other * S.Half)

    def __rpow__(self, other):
        raise NotImplementedError

    @property
    @abstractmethod
    def nocoeff(self):
        raise NotImplementedError('abstract method')

    @property
    @abstractmethod
    def coeff(self):
        raise NotImplementedError('abstract method')

    @abstractmethod
    def get_indices(self):
        raise NotImplementedError('abstract method')

    @abstractmethod
    def get_free_indices(self) -> list[TensorIndex]:
        raise NotImplementedError('abstract method')

    @abstractmethod
    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        raise NotImplementedError('abstract method')

    def fun_eval(self, *index_tuples):
        deprecate_fun_eval()
        return self.substitute_indices(*index_tuples)

    def get_matrix(self):
        """
        DEPRECATED: do not use.

        Returns ndarray components data as a matrix, if components data are
        available and ndarray dimension does not exceed 2.
        """
        from sympy.matrices.dense import Matrix
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if 0 < self.rank <= 2:
                rows = self.data.shape[0]
                columns = self.data.shape[1] if self.rank == 2 else 1
                if self.rank == 2:
                    mat_list = [] * rows
                    for i in range(rows):
                        mat_list.append([])
                        for j in range(columns):
                            mat_list[i].append(self[i, j])
                else:
                    mat_list = [None] * rows
                    for i in range(rows):
                        mat_list[i] = self[i]
                return Matrix(mat_list)
            else:
                raise NotImplementedError('missing multidimensional reduction to matrix.')

    @staticmethod
    def _get_indices_permutation(indices1, indices2):
        return [indices1.index(i) for i in indices2]

    def expand(self, **hints):
        return _expand(self, **hints).doit()

    def _expand(self, **kwargs):
        return self

    def _get_free_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_free_indices_set())
        return indset

    def _get_dummy_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_dummy_indices_set())
        return indset

    def _get_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_indices_set())
        return indset

    @property
    def _iterate_dummy_indices(self):
        dummy_set = self._get_dummy_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in dummy_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos + (p,))
        return recursor(self, ())

    @property
    def _iterate_free_indices(self):
        free_set = self._get_free_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in free_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos + (p,))
        return recursor(self, ())

    @property
    def _iterate_indices(self):

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos + (p,))
        return recursor(self, ())

    @staticmethod
    def _contract_and_permute_with_metric(metric, array, pos, dim):
        from .array import tensorcontraction, tensorproduct, permutedims
        array = tensorcontraction(tensorproduct(metric, array), (1, 2 + pos))
        permu = list(range(dim))
        permu[0], permu[pos] = (permu[pos], permu[0])
        return permutedims(array, permu)

    @staticmethod
    def _match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict):
        from .array import permutedims
        index_types1 = [i.tensor_index_type for i in free_ind1]
        pos2up = []
        pos2down = []
        free2remaining = free_ind2[:]
        for pos1, index1 in enumerate(free_ind1):
            if index1 in free2remaining:
                pos2 = free2remaining.index(index1)
                free2remaining[pos2] = None
                continue
            if -index1 in free2remaining:
                pos2 = free2remaining.index(-index1)
                free2remaining[pos2] = None
                free_ind2[pos2] = index1
                if index1.is_up:
                    pos2up.append(pos2)
                else:
                    pos2down.append(pos2)
            else:
                index2 = free2remaining[pos1]
                if index2 is None:
                    raise ValueError('incompatible indices: %s and %s' % (free_ind1, free_ind2))
                free2remaining[pos1] = None
                free_ind2[pos1] = index1
                if index1.is_up ^ index2.is_up:
                    if index1.is_up:
                        pos2up.append(pos1)
                    else:
                        pos2down.append(pos1)
        if len(set(free_ind1) & set(free_ind2)) < len(free_ind1):
            raise ValueError('incompatible indices: %s and %s' % (free_ind1, free_ind2))
        for pos in pos2up:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError('No metric provided to lower index')
            metric = replacement_dict[index_type_pos]
            metric_inverse = _TensorDataLazyEvaluator.inverse_matrix(metric)
            array = TensExpr._contract_and_permute_with_metric(metric_inverse, array, pos, len(free_ind1))
        for pos in pos2down:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError('No metric provided to lower index')
            metric = replacement_dict[index_type_pos]
            array = TensExpr._contract_and_permute_with_metric(metric, array, pos, len(free_ind1))
        if free_ind1:
            permutation = TensExpr._get_indices_permutation(free_ind2, free_ind1)
            array = permutedims(array, permutation)
        if hasattr(array, 'rank') and array.rank() == 0:
            array = array[()]
        return (free_ind2, array)

    def replace_with_arrays(self, replacement_dict, indices=None):
        """
        Replace the tensorial expressions with arrays. The final array will
        correspond to the N-dimensional array with indices arranged according
        to ``indices``.

        Parameters
        ==========

        replacement_dict
            dictionary containing the replacement rules for tensors.
        indices
            the index order with respect to which the array is read. The
            original index order will be used if no value is passed.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices
        >>> from sympy.tensor.tensor import TensorHead
        >>> from sympy import symbols, diag

        >>> L = TensorIndexType("L")
        >>> i, j = tensor_indices("i j", L)
        >>> A = TensorHead("A", [L])
        >>> A(i).replace_with_arrays({A(i): [1, 2]}, [i])
        [1, 2]

        Since 'indices' is optional, we can also call replace_with_arrays by
        this way if no specific index order is needed:

        >>> A(i).replace_with_arrays({A(i): [1, 2]})
        [1, 2]

        >>> expr = A(i)*A(j)
        >>> expr.replace_with_arrays({A(i): [1, 2]})
        [[1, 2], [2, 4]]

        For contractions, specify the metric of the ``TensorIndexType``, which
        in this case is ``L``, in its covariant form:

        >>> expr = A(i)*A(-i)
        >>> expr.replace_with_arrays({A(i): [1, 2], L: diag(1, -1)})
        -3

        Symmetrization of an array:

        >>> H = TensorHead("H", [L, L])
        >>> a, b, c, d = symbols("a b c d")
        >>> expr = H(i, j)/2 + H(j, i)/2
        >>> expr.replace_with_arrays({H(i, j): [[a, b], [c, d]]})
        [[a, b/2 + c/2], [b/2 + c/2, d]]

        Anti-symmetrization of an array:

        >>> expr = H(i, j)/2 - H(j, i)/2
        >>> repl = {H(i, j): [[a, b], [c, d]]}
        >>> expr.replace_with_arrays(repl)
        [[0, b/2 - c/2], [-b/2 + c/2, 0]]

        The same expression can be read as the transpose by inverting ``i`` and
        ``j``:

        >>> expr.replace_with_arrays(repl, [j, i])
        [[0, -b/2 + c/2], [b/2 - c/2, 0]]
        """
        from .array import Array
        indices = indices or []
        remap = {k.args[0] if k.is_up else -k.args[0]: k for k in self.get_free_indices()}
        for i, index in enumerate(indices):
            if isinstance(index, (Symbol, Mul)):
                if index in remap:
                    indices[i] = remap[index]
                else:
                    indices[i] = -remap[-index]
        replacement_dict = {tensor: Array(array) for tensor, array in replacement_dict.items()}
        for tensor, array in replacement_dict.items():
            if isinstance(tensor, TensorIndexType):
                expected_shape = [tensor.dim for i in range(2)]
            else:
                expected_shape = [index_type.dim for index_type in tensor.index_types]
            if len(expected_shape) != array.rank() or not all((dim1 == dim2 if dim1.is_number else True for dim1, dim2 in zip(expected_shape, array.shape))):
                raise ValueError('shapes for tensor %s expected to be %s, replacement array shape is %s' % (tensor, expected_shape, array.shape))
        ret_indices, array = self._extract_data(replacement_dict)
        last_indices, array = self._match_indices_with_other_tensor(array, indices, ret_indices, replacement_dict)
        return array

    def _check_add_Sum(self, expr, index_symbols):
        from sympy.concrete.summations import Sum
        indices = self.get_indices()
        dum = self.dum
        sum_indices = [(index_symbols[i], 0, indices[i].tensor_index_type.dim - 1) for i, j in dum]
        if sum_indices:
            expr = Sum(expr, *sum_indices)
        return expr

    def _expand_partial_derivative(self):
        return self.func(*[a._expand_partial_derivative() if isinstance(a, TensExpr) else a for a in self.args])