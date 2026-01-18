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
class PermuteDims(_CodegenArrayAbstract):
    """
    Class to represent permutation of axes of arrays.

    Examples
    ========

    >>> from sympy.tensor.array import permutedims
    >>> from sympy import MatrixSymbol
    >>> M = MatrixSymbol("M", 3, 3)
    >>> cg = permutedims(M, [1, 0])

    The object ``cg`` represents the transposition of ``M``, as the permutation
    ``[1, 0]`` will act on its indices by switching them:

    `M_{ij} \\Rightarrow M_{ji}`

    This is evident when transforming back to matrix form:

    >>> from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
    >>> convert_array_to_matrix(cg)
    M.T

    >>> N = MatrixSymbol("N", 3, 2)
    >>> cg = permutedims(N, [1, 0])
    >>> cg.shape
    (2, 3)

    There are optional parameters that can be used as alternative to the permutation:

    >>> from sympy.tensor.array.expressions import ArraySymbol, PermuteDims
    >>> M = ArraySymbol("M", (1, 2, 3, 4, 5))
    >>> expr = PermuteDims(M, index_order_old="ijklm", index_order_new="kijml")
    >>> expr
    PermuteDims(M, (0 2 1)(3 4))
    >>> expr.shape
    (3, 1, 2, 5, 4)

    Permutations of tensor products are simplified in order to achieve a
    standard form:

    >>> from sympy.tensor.array import tensorproduct
    >>> M = MatrixSymbol("M", 4, 5)
    >>> tp = tensorproduct(M, N)
    >>> tp.shape
    (4, 5, 3, 2)
    >>> perm1 = permutedims(tp, [2, 3, 1, 0])

    The args ``(M, N)`` have been sorted and the permutation has been
    simplified, the expression is equivalent:

    >>> perm1.expr.args
    (N, M)
    >>> perm1.shape
    (3, 2, 5, 4)
    >>> perm1.permutation
    (2 3)

    The permutation in its array form has been simplified from
    ``[2, 3, 1, 0]`` to ``[0, 1, 3, 2]``, as the arguments of the tensor
    product `M` and `N` have been switched:

    >>> perm1.permutation.array_form
    [0, 1, 3, 2]

    We can nest a second permutation:

    >>> perm2 = permutedims(perm1, [1, 0, 2, 3])
    >>> perm2.shape
    (2, 3, 5, 4)
    >>> perm2.permutation.array_form
    [1, 0, 3, 2]
    """

    def __new__(cls, expr, permutation=None, index_order_old=None, index_order_new=None, **kwargs):
        from sympy.combinatorics import Permutation
        expr = _sympify(expr)
        expr_rank = get_rank(expr)
        permutation = cls._get_permutation_from_arguments(permutation, index_order_old, index_order_new, expr_rank)
        permutation = Permutation(permutation)
        permutation_size = permutation.size
        if permutation_size != expr_rank:
            raise ValueError('Permutation size must be the length of the shape of expr')
        canonicalize = kwargs.pop('canonicalize', False)
        obj = Basic.__new__(cls, expr, permutation)
        obj._subranks = [get_rank(expr)]
        shape = get_shape(expr)
        if shape is None:
            obj._shape = None
        else:
            obj._shape = tuple((shape[permutation(i)] for i in range(len(shape))))
        if canonicalize:
            return obj._canonicalize()
        return obj

    def _canonicalize(self):
        expr = self.expr
        permutation = self.permutation
        if isinstance(expr, PermuteDims):
            subexpr = expr.expr
            subperm = expr.permutation
            permutation = permutation * subperm
            expr = subexpr
        if isinstance(expr, ArrayContraction):
            expr, permutation = self._PermuteDims_denestarg_ArrayContraction(expr, permutation)
        if isinstance(expr, ArrayTensorProduct):
            expr, permutation = self._PermuteDims_denestarg_ArrayTensorProduct(expr, permutation)
        if isinstance(expr, (ZeroArray, ZeroMatrix)):
            return ZeroArray(*[expr.shape[i] for i in permutation.array_form])
        plist = permutation.array_form
        if plist == sorted(plist):
            return expr
        return self.func(expr, permutation, canonicalize=False)

    @property
    def expr(self):
        return self.args[0]

    @property
    def permutation(self):
        return self.args[1]

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

    @classmethod
    def _PermuteDims_denestarg_ArrayContraction(cls, expr, permutation):
        if not isinstance(expr, ArrayContraction):
            return (expr, permutation)
        if not isinstance(expr.expr, ArrayTensorProduct):
            return (expr, permutation)
        args = expr.expr.args
        subranks = [get_rank(arg) for arg in expr.expr.args]
        contraction_indices = expr.contraction_indices
        contraction_indices_flat = [j for i in contraction_indices for j in i]
        cumul = list(accumulate([0] + subranks))
        permutation_array_blocks_up = []
        image_form = _af_invert(permutation.array_form)
        counter = 0
        for i, e in enumerate(subranks):
            current = []
            for j in range(cumul[i], cumul[i + 1]):
                if j in contraction_indices_flat:
                    continue
                current.append(image_form[counter])
                counter += 1
            permutation_array_blocks_up.append(current)
        index_blocks = [list(range(cumul[i], cumul[i + 1])) for i, e in enumerate(expr.subranks)]
        index_blocks_up = expr._push_indices_up(expr.contraction_indices, index_blocks)
        inverse_permutation = permutation ** (-1)
        index_blocks_up_permuted = [[inverse_permutation(j) for j in i if j is not None] for i in index_blocks_up]
        sorting_keys = list(enumerate(index_blocks_up_permuted))
        sorting_keys.sort(key=lambda x: x[1])
        new_perm_image_form = [i[0] for i in sorting_keys]
        new_index_blocks = [index_blocks[i] for i in new_perm_image_form]
        new_index_perm_array_form = _af_invert([j for i in new_index_blocks for j in i])
        new_args = [args[i] for i in new_perm_image_form]
        new_contraction_indices = [tuple((new_index_perm_array_form[j] for j in i)) for i in contraction_indices]
        new_expr = _array_contraction(_array_tensor_product(*new_args), *new_contraction_indices)
        new_permutation = Permutation(_af_invert([j for i in [permutation_array_blocks_up[k] for k in new_perm_image_form] for j in i]))
        return (new_expr, new_permutation)

    @classmethod
    def _check_permutation_mapping(cls, expr, permutation):
        subranks = expr.subranks
        index2arg = [i for i, arg in enumerate(expr.args) for j in range(expr.subranks[i])]
        permuted_indices = [permutation(i) for i in range(expr.subrank())]
        new_args = list(expr.args)
        arg_candidate_index = index2arg[permuted_indices[0]]
        current_indices = []
        new_permutation = []
        inserted_arg_cand_indices = set()
        for i, idx in enumerate(permuted_indices):
            if index2arg[idx] != arg_candidate_index:
                new_permutation.extend(current_indices)
                current_indices = []
                arg_candidate_index = index2arg[idx]
            current_indices.append(idx)
            arg_candidate_rank = subranks[arg_candidate_index]
            if len(current_indices) == arg_candidate_rank:
                new_permutation.extend(sorted(current_indices))
                local_current_indices = [j - min(current_indices) for j in current_indices]
                i1 = index2arg[i]
                new_args[i1] = _permute_dims(new_args[i1], Permutation(local_current_indices))
                inserted_arg_cand_indices.add(arg_candidate_index)
                current_indices = []
        new_permutation.extend(current_indices)
        args_positions = list(range(len(new_args)))
        maps = {}
        cumulative_subranks = [0] + list(accumulate(subranks))
        for i in range(len(subranks)):
            s = {index2arg[new_permutation[j]] for j in range(cumulative_subranks[i], cumulative_subranks[i + 1])}
            if len(s) != 1:
                continue
            elem = next(iter(s))
            if i != elem:
                maps[i] = elem
        lines = []
        current_line = []
        while maps:
            if len(current_line) == 0:
                k, v = maps.popitem()
                current_line.append(k)
            else:
                k = current_line[-1]
                if k not in maps:
                    current_line = []
                    continue
                v = maps.pop(k)
            if v in current_line:
                lines.append(current_line)
                current_line = []
                continue
            current_line.append(v)
        for line in lines:
            for i, e in enumerate(line):
                args_positions[line[(i + 1) % len(line)]] = e
        permutation_blocks = [[new_permutation[cumulative_subranks[i] + j] for j in range(e)] for i, e in enumerate(subranks)]
        new_args = [new_args[i] for i in args_positions]
        new_permutation_blocks = [permutation_blocks[i] for i in args_positions]
        new_permutation2 = [j for i in new_permutation_blocks for j in i]
        return (_array_tensor_product(*new_args), Permutation(new_permutation2))

    @classmethod
    def _check_if_there_are_closed_cycles(cls, expr, permutation):
        args = list(expr.args)
        subranks = expr.subranks
        cyclic_form = permutation.cyclic_form
        cumulative_subranks = [0] + list(accumulate(subranks))
        cyclic_min = [min(i) for i in cyclic_form]
        cyclic_max = [max(i) for i in cyclic_form]
        cyclic_keep = []
        for i, cycle in enumerate(cyclic_form):
            flag = True
            for j in range(len(cumulative_subranks) - 1):
                if cyclic_min[i] >= cumulative_subranks[j] and cyclic_max[i] < cumulative_subranks[j + 1]:
                    args[j] = _permute_dims(args[j], Permutation([[k - cumulative_subranks[j] for k in cyclic_form[i]]]))
                    flag = False
                    break
            if flag:
                cyclic_keep.append(cyclic_form[i])
        return (_array_tensor_product(*args), Permutation(cyclic_keep, size=permutation.size))

    def nest_permutation(self):
        """
        DEPRECATED.
        """
        ret = self._nest_permutation(self.expr, self.permutation)
        if ret is None:
            return self
        return ret

    @classmethod
    def _nest_permutation(cls, expr, permutation):
        if isinstance(expr, ArrayTensorProduct):
            return _permute_dims(*cls._check_if_there_are_closed_cycles(expr, permutation))
        elif isinstance(expr, ArrayContraction):
            cycles = permutation.cyclic_form
            newcycles = ArrayContraction._convert_outer_indices_to_inner_indices(expr, *cycles)
            newpermutation = Permutation(newcycles)
            new_contr_indices = [tuple((newpermutation(j) for j in i)) for i in expr.contraction_indices]
            return _array_contraction(PermuteDims(expr.expr, newpermutation), *new_contr_indices)
        elif isinstance(expr, ArrayAdd):
            return _array_add(*[PermuteDims(arg, permutation) for arg in expr.args])
        return None

    def as_explicit(self):
        expr = self.expr
        if hasattr(expr, 'as_explicit'):
            expr = expr.as_explicit()
        return permutedims(expr, self.permutation)

    @classmethod
    def _get_permutation_from_arguments(cls, permutation, index_order_old, index_order_new, dim):
        if permutation is None:
            if index_order_new is None or index_order_old is None:
                raise ValueError('Permutation not defined')
            return PermuteDims._get_permutation_from_index_orders(index_order_old, index_order_new, dim)
        else:
            if index_order_new is not None:
                raise ValueError('index_order_new cannot be defined with permutation')
            if index_order_old is not None:
                raise ValueError('index_order_old cannot be defined with permutation')
            return permutation

    @classmethod
    def _get_permutation_from_index_orders(cls, index_order_old, index_order_new, dim):
        if len(set(index_order_new)) != dim:
            raise ValueError('wrong number of indices in index_order_new')
        if len(set(index_order_old)) != dim:
            raise ValueError('wrong number of indices in index_order_old')
        if len(set.symmetric_difference(set(index_order_new), set(index_order_old))) > 0:
            raise ValueError('index_order_new and index_order_old must have the same indices')
        permutation = [index_order_old.index(i) for i in index_order_new]
        return permutation