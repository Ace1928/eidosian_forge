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
class _IndexStructure(CantSympify):
    """
    This class handles the indices (free and dummy ones). It contains the
    algorithms to manage the dummy indices replacements and contractions of
    free indices under multiplications of tensor expressions, as well as stuff
    related to canonicalization sorting, getting the permutation of the
    expression and so on. It also includes tools to get the ``TensorIndex``
    objects corresponding to the given index structure.
    """

    def __init__(self, free, dum, index_types, indices, canon_bp=False):
        self.free = free
        self.dum = dum
        self.index_types = index_types
        self.indices = indices
        self._ext_rank = len(self.free) + 2 * len(self.dum)
        self.dum.sort(key=lambda x: x[0])

    @staticmethod
    def from_indices(*indices):
        """
        Create a new ``_IndexStructure`` object from a list of ``indices``.

        Explanation
        ===========

        ``indices``     ``TensorIndex`` objects, the indices. Contractions are
                        detected upon construction.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure.from_indices(m0, m1, -m1, m3)
        _IndexStructure([(m0, 0), (m3, 3)], [(1, 2)], [Lorentz, Lorentz, Lorentz, Lorentz])
        """
        free, dum = _IndexStructure._free_dum_from_indices(*indices)
        index_types = [i.tensor_index_type for i in indices]
        indices = _IndexStructure._replace_dummy_names(indices, free, dum)
        return _IndexStructure(free, dum, index_types, indices)

    @staticmethod
    def from_components_free_dum(components, free, dum):
        index_types = []
        for component in components:
            index_types.extend(component.index_types)
        indices = _IndexStructure.generate_indices_from_free_dum_index_types(free, dum, index_types)
        return _IndexStructure(free, dum, index_types, indices)

    @staticmethod
    def _free_dum_from_indices(*indices):
        """
        Convert ``indices`` into ``free``, ``dum`` for single component tensor.

        Explanation
        ===========

        ``free``     list of tuples ``(index, pos, 0)``,
                     where ``pos`` is the position of index in
                     the list of indices formed by the component tensors

        ``dum``      list of tuples ``(pos_contr, pos_cov, 0, 0)``

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices,             _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure._free_dum_from_indices(m0, m1, -m1, m3)
        ([(m0, 0), (m3, 3)], [(1, 2)])
        """
        n = len(indices)
        if n == 1:
            return ([(indices[0], 0)], [])
        free = [True] * len(indices)
        index_dict = {}
        dum = []
        for i, index in enumerate(indices):
            name = index.name
            typ = index.tensor_index_type
            contr = index.is_up
            if (name, typ) in index_dict:
                is_contr, pos = index_dict[name, typ]
                if is_contr:
                    if contr:
                        raise ValueError('two equal contravariant indices in slots %d and %d' % (pos, i))
                    else:
                        free[pos] = False
                        free[i] = False
                elif contr:
                    free[pos] = False
                    free[i] = False
                else:
                    raise ValueError('two equal covariant indices in slots %d and %d' % (pos, i))
                if contr:
                    dum.append((i, pos))
                else:
                    dum.append((pos, i))
            else:
                index_dict[name, typ] = (index.is_up, i)
        free = [(index, i) for i, index in enumerate(indices) if free[i]]
        free.sort()
        return (free, dum)

    def get_indices(self):
        """
        Get a list of indices, creating new tensor indices to complete dummy indices.
        """
        return self.indices[:]

    @staticmethod
    def generate_indices_from_free_dum_index_types(free, dum, index_types):
        indices = [None] * (len(free) + 2 * len(dum))
        for idx, pos in free:
            indices[pos] = idx
        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        for pos1, pos2 in dum:
            typ1 = index_types[pos1]
            indname = generate_dummy_name(typ1)
            indices[pos1] = TensorIndex(indname, typ1, True)
            indices[pos2] = TensorIndex(indname, typ1, False)
        return _IndexStructure._replace_dummy_names(indices, free, dum)

    @staticmethod
    def _get_generator_for_dummy_indices(free):
        cdt = defaultdict(int)
        for indx, ipos in free:
            if indx.name.split('_')[0] == indx.tensor_index_type.dummy_name:
                cdt[indx.tensor_index_type] = max(cdt[indx.tensor_index_type], int(indx.name.split('_')[1]) + 1)

        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd
        return dummy_name_gen

    @staticmethod
    def _replace_dummy_names(indices, free, dum):
        dum.sort(key=lambda x: x[0])
        new_indices = list(indices)
        assert len(indices) == len(free) + 2 * len(dum)
        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        for ipos1, ipos2 in dum:
            typ1 = new_indices[ipos1].tensor_index_type
            indname = generate_dummy_name(typ1)
            new_indices[ipos1] = TensorIndex(indname, typ1, True)
            new_indices[ipos2] = TensorIndex(indname, typ1, False)
        return new_indices

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Get a list of free indices.
        """
        free = sorted(self.free, key=lambda x: x[1])
        return [i[0] for i in free]

    def __str__(self):
        return '_IndexStructure({}, {}, {})'.format(self.free, self.dum, self.index_types)

    def __repr__(self):
        return self.__str__()

    def _get_sorted_free_indices_for_canon(self):
        sorted_free = self.free[:]
        sorted_free.sort(key=lambda x: x[0])
        return sorted_free

    def _get_sorted_dum_indices_for_canon(self):
        return sorted(self.dum, key=lambda x: x[0])

    def _get_lexicographically_sorted_index_types(self):
        permutation = self.indices_canon_args()[0]
        index_types = [None] * self._ext_rank
        for i, it in enumerate(self.index_types):
            index_types[permutation(i)] = it
        return index_types

    def _get_lexicographically_sorted_indices(self):
        permutation = self.indices_canon_args()[0]
        indices = [None] * self._ext_rank
        for i, it in enumerate(self.indices):
            indices[permutation(i)] = it
        return indices

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns a ``_IndexStructure`` instance corresponding to the permutation ``g``.

        Explanation
        ===========

        ``g``  permutation corresponding to the tensor in the representation
        used in canonicalization

        ``is_canon_bp``   if True, then ``g`` is the permutation
        corresponding to the canonical form of the tensor
        """
        sorted_free = [i[0] for i in self._get_sorted_free_indices_for_canon()]
        lex_index_types = self._get_lexicographically_sorted_index_types()
        lex_indices = self._get_lexicographically_sorted_indices()
        nfree = len(sorted_free)
        rank = self._ext_rank
        dum = [[None] * 2 for i in range((rank - nfree) // 2)]
        free = []
        index_types = [None] * rank
        indices = [None] * rank
        for i in range(rank):
            gi = g[i]
            index_types[i] = lex_index_types[gi]
            indices[i] = lex_indices[gi]
            if gi < nfree:
                ind = sorted_free[gi]
                assert index_types[i] == sorted_free[gi].tensor_index_type
                free.append((ind, i))
            else:
                j = gi - nfree
                idum, cov = divmod(j, 2)
                if cov:
                    dum[idum][1] = i
                else:
                    dum[idum][0] = i
        dum = [tuple(x) for x in dum]
        return _IndexStructure(free, dum, index_types, indices)

    def indices_canon_args(self):
        """
        Returns ``(g, dummies, msym, v)``, the entries of ``canonicalize``

        See ``canonicalize`` in ``tensor_can.py`` in combinatorics module.
        """
        from sympy.combinatorics.permutations import _af_new
        n = self._ext_rank
        g = [None] * n + [n, n + 1]

        def metric_symmetry_to_msym(metric):
            if metric is None:
                return None
            sym = metric.symmetry
            if sym == TensorSymmetry.fully_symmetric(2):
                return 0
            if sym == TensorSymmetry.fully_symmetric(-2):
                return 1
            return None
        for i, (indx, ipos) in enumerate(self._get_sorted_free_indices_for_canon()):
            g[ipos] = i
        pos = len(self.free)
        j = len(self.free)
        dummies = []
        prev = None
        a = []
        msym = []
        for ipos1, ipos2 in self._get_sorted_dum_indices_for_canon():
            g[ipos1] = j
            g[ipos2] = j + 1
            j += 2
            typ = self.index_types[ipos1]
            if typ != prev:
                if a:
                    dummies.append(a)
                a = [pos, pos + 1]
                prev = typ
                msym.append(metric_symmetry_to_msym(typ.metric))
            else:
                a.extend([pos, pos + 1])
            pos += 2
        if a:
            dummies.append(a)
        return (_af_new(g), dummies, msym)