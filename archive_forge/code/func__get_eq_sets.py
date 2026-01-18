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
def _get_eq_sets(self):
    """Calculates the dictionary for mapping equivalent atoms onto each other.

        Returns:
            dict: with two possible keys:
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to
                    indices of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry
                    operation that maps atom i unto j.
        """
    UNIT = np.eye(3)
    eq_sets, operations = (defaultdict(set), defaultdict(dict))
    symm_ops = [op.rotation_matrix for op in generate_full_symmops(self.symmops, self.tol)]

    def get_clustered_indices():
        indices = cluster_sites(self.centered_mol, self.tol, give_only_index=True)
        out = list(indices[1].values())
        if indices[0] is not None:
            out.append([indices[0]])
        return out
    for index in get_clustered_indices():
        sites = self.centered_mol.cart_coords[index]
        for i, reference in zip(index, sites):
            for op in symm_ops:
                rotated = np.dot(op, sites.T).T
                matched_indices = find_in_coord_list(rotated, reference, self.tol)
                matched_indices = {dict(enumerate(index))[i] for i in matched_indices}
                eq_sets[i] |= matched_indices
                if i not in operations:
                    operations[i] = {j: op.T if j != i else UNIT for j in matched_indices}
                else:
                    for j in matched_indices:
                        if j not in operations[i]:
                            operations[i][j] = op.T if j != i else UNIT
                for j in matched_indices:
                    if j not in operations:
                        operations[j] = {i: op if j != i else UNIT}
                    elif i not in operations[j]:
                        operations[j][i] = op if j != i else UNIT
    return {'eq_sets': eq_sets, 'sym_ops': operations}