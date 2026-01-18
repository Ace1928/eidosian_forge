from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors
def _get_supercell_size(self, s1, s2):
    """
        Returns the supercell size, and whether the supercell should be applied to s1.
        If fu == 1, s1_supercell is returned as true, to avoid ambiguity.
        """
    if self._supercell_size == 'num_sites':
        fu = len(s2) / len(s1)
    elif self._supercell_size == 'num_atoms':
        fu = s2.composition.num_atoms / s1.composition.num_atoms
    elif self._supercell_size == 'volume':
        fu = s2.volume / s1.volume
    elif not isinstance(self._supercell_size, str):
        s1comp, s2comp = (0, 0)
        for el in self._supercell_size:
            el = get_el_sp(el)
            s1comp += s1.composition[el]
            s2comp += s2.composition[el]
        fu = s2comp / s1comp
    else:
        el = get_el_sp(self._supercell_size)
        if el in s2.composition and el in s1.composition:
            fu = s2.composition[el] / s1.composition[el]
        else:
            raise ValueError('Invalid argument for supercell_size.')
    if fu < 2 / 3:
        return (int(round(1 / fu)), False)
    return (int(round(fu)), True)