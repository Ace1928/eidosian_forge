from __future__ import annotations
import logging
import math
import warnings
from fractions import Fraction
from itertools import groupby, product
from math import gcd
from string import ascii_lowercase
from typing import TYPE_CHECKING, Callable, Literal
import numpy as np
from joblib import Parallel, delayed
from monty.dev import requires
from monty.fractions import lcm
from monty.json import MSONable
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.energy_models import SymmetryModel
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.structure_matcher import SpinComparator, StructureMatcher
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.command_line.enumlib_caller import EnumError, EnumlibAdaptor
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core import DummySpecies, Element, Species, Structure, get_el_sp
from pymatgen.core.surface import SlabGenerator
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.icet import IcetSQS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
from pymatgen.transformations.transformation_abc import AbstractTransformation
class CubicSupercellTransformation(AbstractTransformation):
    """A transformation that aims to generate a nearly cubic supercell structure
    from a structure.

    The algorithm solves for a transformation matrix that makes the supercell
    cubic. The matrix must have integer entries, so entries are rounded (in such
    a way that forces the matrix to be non-singular). From the supercell
    resulting from this transformation matrix, vector projections are used to
    determine the side length of the largest cube that can fit inside the
    supercell. The algorithm will iteratively increase the size of the supercell
    until the largest inscribed cube's side length is at least 'min_length'
    and the number of atoms in the supercell falls in the range
    ``min_atoms < n < max_atoms``.
    """

    def __init__(self, min_atoms: int | None=None, max_atoms: int | None=None, min_length: float=15.0, force_diagonal: bool=False, force_90_degrees: bool=False, angle_tolerance: float=0.001):
        """
        Args:
            max_atoms: Maximum number of atoms allowed in the supercell.
            min_atoms: Minimum number of atoms allowed in the supercell.
            min_length: Minimum length of the smallest supercell lattice vector.
            force_diagonal: If True, return a transformation with a diagonal
                transformation matrix.
            force_90_degrees: If True, return a transformation for a supercell
                with 90 degree angles (if possible). To avoid long run times,
                please use max_atoms
            angle_tolerance: tolerance to determine the 90 degree angles.
        """
        self.min_atoms = min_atoms or -np.inf
        self.max_atoms = max_atoms or np.inf
        self.min_length = min_length
        self.force_diagonal = force_diagonal
        self.force_90_degrees = force_90_degrees
        self.angle_tolerance = angle_tolerance
        self.transformation_matrix = None

    def apply_transformation(self, structure: Structure) -> Structure:
        """The algorithm solves for a transformation matrix that makes the
        supercell cubic. The matrix must have integer entries, so entries are
        rounded (in such a way that forces the matrix to be non-singular). From
        the supercell resulting from this transformation matrix, vector
        projections are used to determine the side length of the largest cube
        that can fit inside the supercell. The algorithm will iteratively
        increase the size of the supercell until the largest inscribed cube's
        side length is at least 'num_nn_dists' times the nearest neighbor
        distance and the number of atoms in the supercell falls in the range
        defined by min_atoms and max_atoms.

        Returns:
            supercell: Transformed supercell.
        """
        lat_vecs = structure.lattice.matrix
        sc_not_found = True
        if self.force_diagonal:
            scale = self.min_length / np.array(structure.lattice.abc)
            self.transformation_matrix = np.diag(np.ceil(scale).astype(int))
            st = SupercellTransformation(self.transformation_matrix)
            return st.apply_transformation(structure)
        target_sc_size = self.min_length
        while sc_not_found:
            target_sc_lat_vecs = np.eye(3, 3) * target_sc_size
            self.transformation_matrix = target_sc_lat_vecs @ np.linalg.inv(lat_vecs)
            self.transformation_matrix = _round_and_make_arr_singular(self.transformation_matrix)
            proposed_sc_lat_vecs = self.transformation_matrix @ lat_vecs
            a = proposed_sc_lat_vecs[0]
            b = proposed_sc_lat_vecs[1]
            c = proposed_sc_lat_vecs[2]
            length1_vec = c - _proj(c, a)
            length2_vec = a - _proj(a, c)
            length3_vec = b - _proj(b, a)
            length4_vec = a - _proj(a, b)
            length5_vec = b - _proj(b, c)
            length6_vec = c - _proj(c, b)
            length_vecs = np.array([length1_vec, length2_vec, length3_vec, length4_vec, length5_vec, length6_vec])
            st = SupercellTransformation(self.transformation_matrix)
            superstructure = st.apply_transformation(structure)
            n_atoms = len(superstructure)
            if (np.min(np.linalg.norm(length_vecs, axis=1)) >= self.min_length and self.min_atoms <= n_atoms <= self.max_atoms) and (not self.force_90_degrees or np.all(np.absolute(np.array(superstructure.lattice.angles) - 90) < self.angle_tolerance)):
                return superstructure
            target_sc_size += 0.1
            if n_atoms > self.max_atoms:
                raise AttributeError('While trying to solve for the supercell, the max number of atoms was exceeded. Try lowering the numberof nearest neighbor distances.')
        raise AttributeError('Unable to find cubic supercell')

    @property
    def inverse(self):
        """Returns None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns False."""
        return False