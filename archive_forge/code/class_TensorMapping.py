from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class TensorMapping(collections.abc.MutableMapping):
    """Base class for tensor mappings, which function much like
    a dictionary, but use numpy routines to determine approximate
    equality to keys for getting and setting items.

    This is intended primarily for convenience with things like
    stress-strain pairs and fitting data manipulation. In general,
    it is significantly less robust than a typical hashing
    and should be used with care.
    """

    def __init__(self, tensors: Sequence[Tensor]=(), values: Sequence=(), tol: float=1e-05) -> None:
        """Initialize a TensorMapping.

        Args:
            tensors (Sequence[Tensor], optional): Defaults to (,).
            values (Sequence, optional): Values to be associated with tensors. Defaults to (,).
            tol (float, optional): an absolute tolerance for getting and setting items in the mapping.
                Defaults to 1e-5.

        Raises:
            ValueError: if tensors and values are not the same length
        """
        if len(values) != len(tensors):
            raise ValueError('TensorMapping must be initialized with tensors and values of equivalent length')
        self._tensor_list = list(tensors)
        self._value_list = list(values)
        self.tol = tol

    def __getitem__(self, item):
        index = self._get_item_index(item)
        if index is None:
            raise KeyError(f'{item} not found in mapping.')
        return self._value_list[index]

    def __setitem__(self, key, value) -> None:
        index = self._get_item_index(key)
        if index is None:
            self._tensor_list.append(key)
            self._value_list.append(value)
        else:
            self._value_list[index] = value

    def __delitem__(self, key) -> None:
        index = self._get_item_index(key)
        self._tensor_list.pop(index)
        self._value_list.pop(index)

    def __len__(self) -> int:
        return len(self._tensor_list)

    def __iter__(self):
        yield from self._tensor_list

    def values(self):
        """Values in mapping."""
        return self._value_list

    def items(self):
        """Items in mapping."""
        return zip(self._tensor_list, self._value_list)

    def __contains__(self, item) -> bool:
        return self._get_item_index(item) is not None

    def _get_item_index(self, item):
        if len(self._tensor_list) == 0:
            return None
        item = np.array(item)
        axis = tuple(range(1, len(item.shape) + 1))
        mask = np.all(np.abs(np.array(self._tensor_list) - item) < self.tol, axis=axis)
        indices = np.where(mask)[0]
        if len(indices) > 1:
            raise ValueError('Tensor key collision.')
        if len(indices) == 0:
            return None
        return indices[0]