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
@classmethod
def _get_reduced_structure(cls, struct: Structure, primitive_cell: bool=True, niggli: bool=True) -> Structure:
    """Helper method to find a reduced structure."""
    reduced = struct.copy()
    if niggli:
        reduced = reduced.get_reduced_structure(reduction_algo='niggli')
    if primitive_cell:
        reduced = reduced.get_primitive_structure()
    return reduced