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
class WildTensor(Tensor):
    """
    A wild object which matches ``Tensor`` instances

    Explanation
    ===========
    This is instantiated by attaching indices to a ``WildTensorHead`` instance.

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)
    >>> K = TensorHead('K', [R3])
    >>> Q = TensorHead('Q', [R3, R3])

    Matching also takes the indices into account
    >>> W(p).matches(K(p))
    {W(R3): _WildTensExpr(K(p))}
    >>> W(p).matches(K(q))
    >>> W(p).matches(K(-p))

    If you want to match objects with any number of indices, just use a ``WildTensor`` with no indices.
    >>> W().matches(K(p))
    {W: K(p)}
    >>> W().matches(Q(p,q))
    {W: Q(p, q)}

    See Also
    ========
    ``WildTensorHead``
    ``Tensor``

    """

    def __new__(cls, tensor_head, indices, **kw_args):
        is_canon_bp = kw_args.pop('is_canon_bp', False)
        if tensor_head.func == TensorHead:
            '\n            If someone tried to call WildTensor by supplying a TensorHead (not a WildTensorHead), return a normal tensor instead. This is helpful when using subs on an expression to replace occurrences of a WildTensorHead with a TensorHead.\n            '
            return Tensor(tensor_head, indices, is_canon_bp=is_canon_bp, **kw_args)
        elif tensor_head.func == _WildTensExpr:
            return tensor_head(*indices)
        indices = cls._parse_indices(tensor_head, indices)
        index_types = [ind.tensor_index_type for ind in indices]
        tensor_head = tensor_head.func(tensor_head.name, index_types, symmetry=None, comm=tensor_head.comm, unordered_indices=tensor_head.unordered_indices)
        obj = Basic.__new__(cls, tensor_head, Tuple(*indices))
        obj.name = tensor_head.name
        obj._index_structure = _IndexStructure.from_indices(*indices)
        obj._free = obj._index_structure.free[:]
        obj._dum = obj._index_structure.dum[:]
        obj._ext_rank = obj._index_structure._ext_rank
        obj._coeff = S.One
        obj._nocoeff = obj
        obj._component = tensor_head
        obj._components = [tensor_head]
        if tensor_head.rank != len(indices):
            raise ValueError('wrong number of indices')
        obj.is_canon_bp = is_canon_bp
        obj._index_map = obj._build_index_map(indices, obj._index_structure)
        return obj

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, TensExpr) and expr != S(1):
            return None
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()
        if len(self.indices) > 0:
            if not hasattr(expr, 'get_free_indices'):
                return None
            expr_indices = expr.get_free_indices()
            if len(expr_indices) != len(self.indices):
                return None
            if self._component.unordered_indices:
                m = self._match_indices_ignoring_order(expr)
                if m is None:
                    return None
                else:
                    repl_dict.update(m)
            else:
                for i in range(len(expr_indices)):
                    m = self.indices[i].matches(expr_indices[i])
                    if m is None:
                        return None
                    else:
                        repl_dict.update(m)
            repl_dict[self.component] = _WildTensExpr(expr)
        else:
            repl_dict[self] = expr
        return repl_dict

    def _match_indices_ignoring_order(self, expr, repl_dict=None, old=False):
        """
        Helper method for matches. Checks if the indices of self and expr
        match disregarding index ordering.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        def siftkey(ind):
            if isinstance(ind, WildTensorIndex):
                if ind.ignore_updown:
                    return 'wild, updown'
                else:
                    return 'wild'
            else:
                return 'nonwild'
        indices_sifted = sift(self.indices, siftkey)
        matched_indices = []
        expr_indices_remaining = expr.get_indices()
        for ind in indices_sifted['nonwild']:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                if e_ind in matched_indices:
                    continue
                m = ind.matches(e_ind)
                if m is not None:
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None
        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
        for ind in indices_sifted['wild']:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                m = ind.matches(e_ind)
                if m is not None:
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None
        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
        for ind in indices_sifted['wild, updown']:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                m = ind.matches(e_ind)
                if m is not None:
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None
        if len(matched_indices) < len(self.indices):
            return None
        else:
            return repl_dict