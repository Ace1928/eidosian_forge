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
class TensMul(TensExpr, AssocOp):
    """
    Product of tensors.

    Parameters
    ==========

    coeff : SymPy coefficient of the tensor
    args

    Attributes
    ==========

    ``components`` : list of ``TensorHead`` of the component tensors
    ``types`` : list of nonrepeated ``TensorIndexType``
    ``free`` : list of ``(ind, ipos, icomp)``, see Notes
    ``dum`` : list of ``(ipos1, ipos2, icomp1, icomp2)``, see Notes
    ``ext_rank`` : rank of the tensor counting the dummy indices
    ``rank`` : rank of the tensor
    ``coeff`` : SymPy coefficient of the tensor
    ``free_args`` : list of the free indices in sorted order
    ``is_canon_bp`` : ``True`` if the tensor in in canonical form

    Notes
    =====

    ``args[0]``   list of ``TensorHead`` of the component tensors.

    ``args[1]``   list of ``(ind, ipos, icomp)``
    where ``ind`` is a free index, ``ipos`` is the slot position
    of ``ind`` in the ``icomp``-th component tensor.

    ``args[2]`` list of tuples representing dummy indices.
    ``(ipos1, ipos2, icomp1, icomp2)`` indicates that the contravariant
    dummy index is the ``ipos1``-th slot position in the ``icomp1``-th
    component tensor; the corresponding covariant index is
    in the ``ipos2`` slot position in the ``icomp2``-th component tensor.

    """
    identity = S.One
    _index_structure = None

    def __new__(cls, *args, **kw_args):
        is_canon_bp = kw_args.get('is_canon_bp', False)
        args = list(map(_sympify, args))
        '\n        If the internal dummy indices in one arg conflict with the free indices\n        of the remaining args, we need to rename those internal dummy indices.\n        '
        free = [get_free_indices(arg) for arg in args]
        free = set(itertools.chain(*free))
        newargs = []
        for arg in args:
            dum_this = set(get_dummy_indices(arg))
            dum_other = [get_dummy_indices(a) for a in newargs]
            dum_other = set(itertools.chain(*dum_other))
            free_this = set(get_free_indices(arg))
            if len(dum_this.intersection(free)) > 0:
                exclude = free_this.union(free, dum_other)
                newarg = TensMul._dedupe_indices(arg, exclude)
            else:
                newarg = arg
            newargs.append(newarg)
        args = newargs
        args = [i for arg in args for i in (arg.args if isinstance(arg, (TensMul, Mul)) else [arg])]
        args, indices, free, dum = TensMul._tensMul_contract_indices(args, replace_indices=False)
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)
        obj = TensExpr.__new__(cls, *args)
        obj._indices = indices
        obj._index_types = index_types[:]
        obj._index_structure = index_structure
        obj._free = index_structure.free[:]
        obj._dum = index_structure.dum[:]
        obj._free_indices = {x[0] for x in obj.free}
        obj._rank = len(obj.free)
        obj._ext_rank = len(obj._index_structure.free) + 2 * len(obj._index_structure.dum)
        obj._coeff = S.One
        obj._is_canon_bp = is_canon_bp
        return obj
    index_types = property(lambda self: self._index_types)
    free = property(lambda self: self._free)
    dum = property(lambda self: self._dum)
    free_indices = property(lambda self: self._free_indices)
    rank = property(lambda self: self._rank)
    ext_rank = property(lambda self: self._ext_rank)

    @staticmethod
    def _indices_to_free_dum(args_indices):
        free2pos1 = {}
        free2pos2 = {}
        dummy_data = []
        indices = []
        pos2 = 0
        for pos1, arg_indices in enumerate(args_indices):
            for index_pos, index in enumerate(arg_indices):
                if not isinstance(index, TensorIndex):
                    raise TypeError('expected TensorIndex')
                if -index in free2pos1:
                    other_pos1 = free2pos1.pop(-index)
                    other_pos2 = free2pos2.pop(-index)
                    if index.is_up:
                        dummy_data.append((index, pos1, other_pos1, pos2, other_pos2))
                    else:
                        dummy_data.append((-index, other_pos1, pos1, other_pos2, pos2))
                    indices.append(index)
                elif index in free2pos1:
                    raise ValueError('Repeated index: %s' % index)
                else:
                    free2pos1[index] = pos1
                    free2pos2[index] = pos2
                    indices.append(index)
                pos2 += 1
        free = [(i, p) for i, p in free2pos2.items()]
        free_names = [i.name for i in free2pos2.keys()]
        dummy_data.sort(key=lambda x: x[3])
        return (indices, free, free_names, dummy_data)

    @staticmethod
    def _dummy_data_to_dum(dummy_data):
        return [(p2a, p2b) for i, p1a, p1b, p2a, p2b in dummy_data]

    @staticmethod
    def _tensMul_contract_indices(args, replace_indices=True):
        replacements = [{} for _ in args]
        args_indices = [get_indices(arg) for arg in args]
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)
        cdt = defaultdict(int)

        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd
        if replace_indices:
            for old_index, pos1cov, pos1contra, pos2cov, pos2contra in dummy_data:
                index_type = old_index.tensor_index_type
                while True:
                    dummy_name = dummy_name_gen(index_type)
                    if dummy_name not in free_names:
                        break
                dummy = TensorIndex(dummy_name, index_type, True)
                replacements[pos1cov][old_index] = dummy
                replacements[pos1contra][-old_index] = -dummy
                indices[pos2cov] = dummy
                indices[pos2contra] = -dummy
            args = [arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg, repl in zip(args, replacements)]
        dum = TensMul._dummy_data_to_dum(dummy_data)
        return (args, indices, free, dum)

    @staticmethod
    def _get_components_from_args(args):
        """
        Get a list of ``Tensor`` objects having the same ``TIDS`` if multiplied
        by one another.
        """
        components = []
        for arg in args:
            if not isinstance(arg, TensExpr):
                continue
            if isinstance(arg, TensAdd):
                continue
            components.extend(arg.components)
        return components

    @staticmethod
    def _rebuild_tensors_list(args, index_structure):
        indices = index_structure.get_indices()
        ind_pos = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, TensExpr):
                continue
            prev_pos = ind_pos
            ind_pos += arg.ext_rank
            args[i] = Tensor(arg.component, indices[prev_pos:ind_pos])

    def doit(self, **hints):
        is_canon_bp = self._is_canon_bp
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
            "\n            There may now be conflicts between dummy indices of different args\n            (each arg's doit method does not have any information about which\n            dummy indices are already used in the other args), so we\n            deduplicate them.\n            "
            rule = dict(zip(self.args, args))
            rule = self._dedupe_indices_in_rule(rule)
            args = [rule[a] for a in self.args]
        else:
            args = self.args
        args = [arg for arg in args if arg != self.identity]
        coeff = reduce(lambda a, b: a * b, [arg for arg in args if not isinstance(arg, TensExpr)], S.One)
        args = [arg for arg in args if isinstance(arg, TensExpr)]
        if len(args) == 0:
            return coeff
        if coeff != self.identity:
            args = [coeff] + args
        if coeff == 0:
            return S.Zero
        if len(args) == 1:
            return args[0]
        args, indices, free, dum = TensMul._tensMul_contract_indices(args)
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)
        obj = self.func(*args)
        obj._index_types = index_types
        obj._index_structure = index_structure
        obj._ext_rank = len(obj._index_structure.free) + 2 * len(obj._index_structure.dum)
        obj._coeff = coeff
        obj._is_canon_bp = is_canon_bp
        return obj

    @staticmethod
    def from_data(coeff, components, free, dum, **kw_args):
        return TensMul(coeff, *TensMul._get_tensors_from_components_free_dum(components, free, dum), **kw_args).doit()

    @staticmethod
    def _get_tensors_from_components_free_dum(components, free, dum):
        """
        Get a list of ``Tensor`` objects by distributing ``free`` and ``dum`` indices on the ``components``.
        """
        index_structure = _IndexStructure.from_components_free_dum(components, free, dum)
        indices = index_structure.get_indices()
        tensors = [None for i in components]
        ind_pos = 0
        for i, component in enumerate(components):
            prev_pos = ind_pos
            ind_pos += component.rank
            tensors[i] = Tensor(component, indices[prev_pos:ind_pos])
        return tensors

    def _get_free_indices_set(self):
        return {i[0] for i in self.free}

    def _get_dummy_indices_set(self):
        dummy_pos = set(itertools.chain(*self.dum))
        return {idx for i, idx in enumerate(self._index_structure.get_indices()) if i in dummy_pos}

    def _get_position_offset_for_indices(self):
        arg_offset = [None for i in range(self.ext_rank)]
        counter = 0
        for i, arg in enumerate(self.args):
            if not isinstance(arg, TensExpr):
                continue
            for j in range(arg.ext_rank):
                arg_offset[j + counter] = counter
            counter += arg.ext_rank
        return arg_offset

    @property
    def free_args(self):
        return sorted([x[0] for x in self.free])

    @property
    def components(self):
        return self._get_components_from_args(self.args)

    @property
    def free_in_args(self):
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        return [(ind, pos - arg_offset[pos], argpos[pos]) for ind, pos in self.free]

    @property
    def coeff(self):
        return self._coeff

    @property
    def nocoeff(self):
        return self.func(*[t for t in self.args if isinstance(t, TensExpr)]).doit()

    @property
    def dum_in_args(self):
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        return [(p1 - arg_offset[p1], p2 - arg_offset[p2], argpos[p1], argpos[p2]) for p1, p2 in self.dum]

    def equals(self, other):
        if other == 0:
            return self.coeff == 0
        other = _sympify(other)
        if not isinstance(other, TensExpr):
            assert not self.components
            return self.coeff == other
        return self.canon_bp() == other.canon_bp()

    def get_indices(self):
        """
        Returns the list of indices of the tensor.

        Explanation
        ===========

        The indices are listed in the order in which they appear in the
        component tensors.
        The dummy indices are given a name which does not collide with
        the names of the free indices.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_indices()
        [L_0, -L_0, m2]
        """
        return self._indices

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Returns the list of free indices of the tensor.

        Explanation
        ===========

        The indices are listed in the order in which they appear in the
        component tensors.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_free_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_free_indices()
        [m2]
        """
        return self._index_structure.get_free_indices()

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        return self.func(*[arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args])

    def split(self):
        """
        Returns a list of tensors, whose product is ``self``.

        Explanation
        ===========

        Dummy indices contracted among different tensor components
        become free indices with the same name as the one used to
        represent the dummy indices.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
        >>> A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        >>> t = A(a,b)*B(-b,c)
        >>> t
        A(a, L_0)*B(-L_0, c)
        >>> t.split()
        [A(a, L_0), B(-L_0, c)]
        """
        if self.args == ():
            return [self]
        splitp = []
        res = 1
        for arg in self.args:
            if isinstance(arg, Tensor):
                splitp.append(res * arg)
                res = 1
            else:
                res *= arg
        return splitp

    def _expand(self, **hints):
        args = [_expand(arg, **hints) for arg in self.args]
        args1 = [arg.args if isinstance(arg, (Add, TensAdd)) else (arg,) for arg in args]
        return TensAdd(*[TensMul(*i) for i in itertools.product(*args1)])

    def __neg__(self):
        return TensMul(S.NegativeOne, self, is_canon_bp=self._is_canon_bp).doit()

    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def _get_args_for_traditional_printer(self):
        args = list(self.args)
        if self.coeff.could_extract_minus_sign():
            sign = '-'
            if args[0] == S.NegativeOne:
                args = args[1:]
            else:
                args[0] = -args[0]
        else:
            sign = ''
        return (sign, args)

    def _sort_args_for_sorted_components(self):
        """
        Returns the ``args`` sorted according to the components commutation
        properties.

        Explanation
        ===========

        The sorting is done taking into account the commutation group
        of the component tensors.
        """
        cv = [arg for arg in self.args if isinstance(arg, TensExpr)]
        sign = 1
        n = len(cv) - 1
        for i in range(n):
            for j in range(n, i, -1):
                c = cv[j - 1].commutes_with(cv[j])
                if c not in (0, 1):
                    continue
                typ1 = sorted(set(cv[j - 1].component.index_types), key=lambda x: x.name)
                typ2 = sorted(set(cv[j].component.index_types), key=lambda x: x.name)
                if (typ1, cv[j - 1].component.name) > (typ2, cv[j].component.name):
                    cv[j - 1], cv[j] = (cv[j], cv[j - 1])
                    if c:
                        sign = -sign
        coeff = sign * self.coeff
        if coeff != 1:
            return [coeff] + cv
        return cv

    def sorted_components(self):
        """
        Returns a tensor product with sorted components.
        """
        return TensMul(*self._sort_args_for_sorted_components()).doit()

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns the tensor corresponding to the permutation ``g``

        For further details, see the method in ``TIDS`` with the same name.
        """
        return perm2tensor(self, g, is_canon_bp=is_canon_bp)

    def canon_bp(self):
        """
        Canonicalize using the Butler-Portugal algorithm for canonicalization
        under monoterm symmetries.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        >>> t = A(m0,-m1)*A(m1,-m0)
        >>> t.canon_bp()
        -A(L_0, L_1)*A(-L_0, -L_1)
        >>> t = A(m0,-m1)*A(m1,-m2)*A(m2,-m0)
        >>> t.canon_bp()
        0
        """
        if self._is_canon_bp:
            return self
        expr = self.expand()
        if isinstance(expr, TensAdd):
            return expr.canon_bp()
        if not expr.components:
            return expr
        t = expr.sorted_components()
        g, dummies, msym = t._index_structure.indices_canon_args()
        v = components_canon_args(t.components)
        can = canonicalize(g, dummies, msym, *v)
        if can == 0:
            return S.Zero
        tmul = t.perm2tensor(can, True)
        return tmul

    def contract_delta(self, delta):
        t = self.contract_metric(delta)
        return t

    def _get_indices_to_args_pos(self):
        """
        Get a dict mapping the index position to TensMul's argument number.
        """
        pos_map = {}
        pos_counter = 0
        for arg_i, arg in enumerate(self.args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            for i in range(arg.ext_rank):
                pos_map[pos_counter] = arg_i
                pos_counter += 1
        return pos_map

    def contract_metric(self, g):
        """
        Raise or lower indices with the metric ``g``.

        Parameters
        ==========

        g : metric

        Notes
        =====

        See the ``TensorIndexType`` docstring for the contraction conventions.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m0)*q(m1)*g(-m0, -m1)
        >>> t.canon_bp()
        metric(L_0, L_1)*p(-L_0)*q(-L_1)
        >>> t.contract_metric(g).canon_bp()
        p(L_0)*q(-L_0)
        """
        expr = self.expand()
        if self != expr:
            expr = canon_bp(expr)
            return contract_metric(expr, g)
        pos_map = self._get_indices_to_args_pos()
        args = list(self.args)
        if g.symmetry == TensorSymmetry.fully_symmetric(-2):
            antisym = 1
        elif g.symmetry == TensorSymmetry.fully_symmetric(2):
            antisym = 0
        elif g.symmetry == TensorSymmetry.no_symmetry(2):
            antisym = None
        else:
            raise NotImplementedError
        gpos = [i for i, x in enumerate(self.args) if isinstance(x, Tensor) and x.component == g]
        if not gpos:
            return self
        sign = 1
        dum = self.dum[:]
        free = self.free[:]
        elim = set()
        for gposx in gpos:
            if gposx in elim:
                continue
            free1 = [x for x in free if pos_map[x[1]] == gposx]
            dum1 = [x for x in dum if pos_map[x[0]] == gposx or pos_map[x[1]] == gposx]
            if not dum1:
                continue
            elim.add(gposx)
            args[gposx] = 1
            if len(dum1) == 2:
                if not antisym:
                    dum10, dum11 = dum1
                    if pos_map[dum10[1]] == gposx:
                        p0 = dum10[0]
                    else:
                        p0 = dum10[1]
                    if pos_map[dum11[1]] == gposx:
                        p1 = dum11[0]
                    else:
                        p1 = dum11[1]
                    dum.append((p0, p1))
                else:
                    dum10, dum11 = dum1
                    if pos_map[dum10[1]] == gposx:
                        p0 = dum10[0]
                        if dum10[1] == 1:
                            sign = -sign
                    else:
                        p0 = dum10[1]
                        if dum10[0] == 0:
                            sign = -sign
                    if pos_map[dum11[1]] == gposx:
                        p1 = dum11[0]
                        sign = -sign
                    else:
                        p1 = dum11[1]
                    dum.append((p0, p1))
            elif len(dum1) == 1:
                if not antisym:
                    dp0, dp1 = dum1[0]
                    if pos_map[dp0] == pos_map[dp1]:
                        typ = g.index_types[0]
                        sign = sign * typ.dim
                    else:
                        if pos_map[dp0] == gposx:
                            p1 = dp1
                        else:
                            p1 = dp0
                        ind, p = free1[0]
                        free.append((ind, p1))
                else:
                    dp0, dp1 = dum1[0]
                    if pos_map[dp0] == pos_map[dp1]:
                        typ = g.index_types[0]
                        sign = sign * typ.dim
                        if dp0 < dp1:
                            sign = -sign
                    else:
                        if pos_map[dp0] == gposx:
                            p1 = dp1
                            if dp0 == 0:
                                sign = -sign
                        else:
                            p1 = dp0
                        ind, p = free1[0]
                        free.append((ind, p1))
            dum = [x for x in dum if x not in dum1]
            free = [x for x in free if x not in free1]
        shift = 0
        shifts = [0] * len(args)
        for i in range(len(args)):
            if i in elim:
                shift += 2
                continue
            shifts[i] = shift
        free = [(ind, p - shifts[pos_map[p]]) for ind, p in free if pos_map[p] not in elim]
        dum = [(p0 - shifts[pos_map[p0]], p1 - shifts[pos_map[p1]]) for i, (p0, p1) in enumerate(dum) if pos_map[p0] not in elim and pos_map[p1] not in elim]
        res = sign * TensMul(*args).doit()
        if not isinstance(res, TensExpr):
            return res
        im = _IndexStructure.from_components_free_dum(res.components, free, dum)
        return res._set_new_index_structure(im)

    def _set_new_index_structure(self, im, is_canon_bp=False):
        indices = im.get_indices()
        return self._set_indices(*indices, is_canon_bp=is_canon_bp)

    def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
        if len(indices) != self.ext_rank:
            raise ValueError('indices length mismatch')
        args = list(self.args)[:]
        pos = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            ext_rank = arg.ext_rank
            args[i] = arg._set_indices(*indices[pos:pos + ext_rank])
            pos += ext_rank
        return TensMul(*args, is_canon_bp=is_canon_bp).doit()

    @staticmethod
    def _index_replacement_for_contract_metric(args, free, dum):
        for arg in args:
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)

    def substitute_indices(self, *index_tuples):
        new_args = []
        for arg in self.args:
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            new_args.append(arg)
        return TensMul(*new_args).doit()

    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        t = self.substitute_indices(*list(zip(free_args, indices)))
        if len({i if i.is_up else -i for i in indices}) != len(indices):
            return t.func(*t.args)
        return t

    def _extract_data(self, replacement_dict):
        args_indices, arrays = zip(*[arg._extract_data(replacement_dict) for arg in self.args if isinstance(arg, TensExpr)])
        coeff = reduce(operator.mul, [a for a in self.args if not isinstance(a, TensExpr)], S.One)
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)
        dum = TensMul._dummy_data_to_dum(dummy_data)
        ext_rank = self.ext_rank
        free.sort(key=lambda x: x[1])
        free_indices = [i[0] for i in free]
        return (free_indices, coeff * _TensorDataLazyEvaluator.data_contract_dum(arrays, dum, ext_rank))

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            dat = _tensor_data_substitution_dict[self.expand()]
        return dat

    @data.setter
    def data(self, data):
        deprecate_data()
        raise ValueError('Not possible to set component data to a tensor expression')

    @data.deleter
    def data(self):
        deprecate_data()
        raise ValueError('Not possible to delete component data to a tensor expression')

    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError('No iteration on abstract tensors')
            return self.data.__iter__()

    @staticmethod
    def _dedupe_indices(new, exclude):
        """
        exclude: set
        new: TensExpr

        If ``new`` has any dummy indices that are in ``exclude``, return a version
        of new with those indices replaced. If no replacements are needed,
        return None

        """
        exclude = set(exclude)
        dums_new = set(get_dummy_indices(new))
        free_new = set(get_free_indices(new))
        conflicts = dums_new.intersection(exclude)
        if len(conflicts) == 0:
            return None
        '\n        ``exclude_for_gen`` is to be passed to ``_IndexStructure._get_generator_for_dummy_indices()``.\n        Since the latter does not use the index position for anything, we just\n        set it as ``None`` here.\n        '
        exclude.update(dums_new)
        exclude.update(free_new)
        exclude_for_gen = [(i, None) for i in exclude]
        gen = _IndexStructure._get_generator_for_dummy_indices(exclude_for_gen)
        repl = {}
        for d in conflicts:
            if -d in repl.keys():
                continue
            newname = gen(d.tensor_index_type)
            new_d = d.func(newname, *d.args[1:])
            repl[d] = new_d
            repl[-d] = -new_d
        if len(repl) == 0:
            return None
        new_renamed = new._replace_indices(repl)
        return new_renamed

    def _dedupe_indices_in_rule(self, rule):
        """
        rule: dict

        This applies TensMul._dedupe_indices on all values of rule.

        """
        index_rules = {k: v for k, v in rule.items() if isinstance(k, TensorIndex)}
        other_rules = {k: v for k, v in rule.items() if k not in index_rules.keys()}
        exclude = set(self.get_indices())
        newrule = {}
        newrule.update(index_rules)
        exclude.update(index_rules.keys())
        exclude.update(index_rules.values())
        for old, new in other_rules.items():
            new_renamed = TensMul._dedupe_indices(new, exclude)
            if old == new or new_renamed is None:
                newrule[old] = new
            else:
                newrule[old] = new_renamed
                exclude.update(get_indices(new_renamed))
        return newrule

    def _eval_rewrite_as_Indexed(self, *args):
        from sympy.concrete.summations import Sum
        index_symbols = [i.args[0] for i in self.get_indices()]
        args = [arg.args[0] if isinstance(arg, Sum) else arg for arg in args]
        expr = Mul.fromiter(args)
        return self._check_add_Sum(expr, index_symbols)

    def _eval_partial_derivative(self, s):
        terms = []
        for i, arg in enumerate(self.args):
            if isinstance(arg, TensExpr):
                d = arg._eval_partial_derivative(s)
            elif s._diff_wrt:
                d = arg._eval_derivative(s)
            else:
                d = S.Zero
            if d:
                terms.append(TensMul.fromiter(self.args[:i] + (d,) + self.args[i + 1:]))
        return TensAdd.fromiter(terms)