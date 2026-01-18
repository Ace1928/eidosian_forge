from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
@staticmethod
def _combine_eq_sets(equiv_sets, sym_ops):
    """Combines the dicts of _get_equivalent_atom_dicts into one.

        Args:
            equiv_sets (dict): Map of equivalent atoms onto each other (i.e. indices to indices).
            sym_ops (dict): Map of symmetry operations that map atoms onto each other.

        Returns:
            dict: with two possible keys:
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to
                    indices of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry
                    operation that maps atom i unto j.
        """
    unit_mat = np.eye(3)

    def all_equivalent_atoms_of_i(idx, eq_sets, ops):
        """WORKS INPLACE on operations."""
        visited = {idx}
        tmp_eq_sets = {j: eq_sets[j] - visited for j in eq_sets[idx]}
        while tmp_eq_sets:
            new_tmp_eq_sets = {}
            for j in tmp_eq_sets:
                if j in visited:
                    continue
                visited.add(j)
                for k in tmp_eq_sets[j]:
                    new_tmp_eq_sets[k] = eq_sets[k] - visited
                    if idx not in ops[k]:
                        ops[k][idx] = np.dot(ops[j][idx], ops[k][j]) if k != idx else unit_mat
                    ops[idx][k] = ops[k][idx].T
            tmp_eq_sets = new_tmp_eq_sets
        return (visited, ops)
    equiv_sets = copy.deepcopy(equiv_sets)
    ops = copy.deepcopy(sym_ops)
    to_be_deleted = set()
    for idx in equiv_sets:
        if idx in to_be_deleted:
            continue
        visited, ops = all_equivalent_atoms_of_i(idx, equiv_sets, ops)
        to_be_deleted |= visited - {idx}
    for key in to_be_deleted:
        equiv_sets.pop(key, None)
    return {'eq_sets': equiv_sets, 'sym_ops': ops}