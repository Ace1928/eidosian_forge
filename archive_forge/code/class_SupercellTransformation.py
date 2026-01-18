from __future__ import annotations
import logging
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from numpy import around
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, get_el_sp
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.site_transformations import PartialRemoveSitesTransformation
from pymatgen.transformations.transformation_abc import AbstractTransformation
class SupercellTransformation(AbstractTransformation):
    """The SupercellTransformation replicates a unit cell to a supercell."""

    def __init__(self, scaling_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        """
        Args:
            scaling_matrix: A matrix of transforming the lattice vectors.
                Defaults to the identity matrix. Has to be all integers. e.g.,
                [[2,1,0],[0,3,0],[0,0,1]] generates a new structure with
                lattice vectors a" = 2a + b, b" = 3b, c" = c where a, b, and c
                are the lattice vectors of the original structure.
        """
        self.scaling_matrix = scaling_matrix

    @classmethod
    def from_scaling_factors(cls, scale_a: float=1, scale_b: float=1, scale_c: float=1) -> Self:
        """Convenience method to get a SupercellTransformation from a simple
        series of three numbers for scaling each lattice vector. Equivalent to
        calling the normal with [[scale_a, 0, 0], [0, scale_b, 0],
        [0, 0, scale_c]].

        Args:
            scale_a: Scaling factor for lattice direction a. Defaults to 1.
            scale_b: Scaling factor for lattice direction b. Defaults to 1.
            scale_c: Scaling factor for lattice direction c. Defaults to 1.

        Returns:
            SupercellTransformation.
        """
        return cls([[scale_a, 0, 0], [0, scale_b, 0], [0, 0, scale_c]])

    @classmethod
    def from_boundary_distance(cls, structure: Structure, min_boundary_dist: float=6, allow_rotation: bool=False, max_atoms: float=-1) -> Self:
        """Get a SupercellTransformation according to the desired minimum distance between periodic
        boundaries of the resulting supercell.

        Args:
            structure (Structure): Input structure.
            min_boundary_dist (float): Desired minimum distance between all periodic boundaries. Defaults to 6.
            allow_rotation (bool): Whether allowing lattice angles to change. Only useful when
                at least two of the three lattice vectors are required to expand. Defaults to False.
                If True, a SupercellTransformation satisfying min_boundary_dist but with smaller
                number of atoms than the SupercellTransformation with unchanged lattice angles
                can possibly be found. If such a SupercellTransformation cannot be found easily,
                the SupercellTransformation with unchanged lattice angles will be returned.
            max_atoms (int): Maximum number of atoms allowed in the supercell. Defaults to -1 for infinity.

        Returns:
            SupercellTransformation.
        """
        min_expand = np.int8(min_boundary_dist / np.array([structure.lattice.d_hkl(plane) for plane in np.eye(3)]))
        max_atoms = max_atoms if max_atoms > 0 else float('inf')
        if allow_rotation and sum(min_expand != 0) > 1:
            min1, min2, min3 = map(int, min_expand)
            scaling_matrix = [[min1 or 1, 1 if min1 and min2 else 0, 1 if min1 and min3 else 0], [-1 if min2 and min1 else 0, min2 or 1, 1 if min2 and min3 else 0], [-1 if min3 and min1 else 0, -1 if min3 and min2 else 0, min3 or 1]]
            struct_scaled = structure.make_supercell(scaling_matrix, in_place=False)
            min_expand_scaled = np.int8(min_boundary_dist / np.array([struct_scaled.lattice.d_hkl(plane) for plane in np.eye(3)]))
            if sum(min_expand_scaled != 0) == 0 and len(struct_scaled) <= max_atoms:
                return cls(scaling_matrix)
        scaling_matrix = np.eye(3) + np.diag(min_expand)
        struct_scaled = structure.make_supercell(scaling_matrix, in_place=False)
        if len(struct_scaled) <= max_atoms:
            return cls(scaling_matrix)
        msg = f'max_atoms={max_atoms!r} exceeded while trying to solve for supercell. You can try lowering min_boundary_dist={min_boundary_dist!r}'
        if not allow_rotation:
            msg += ' or set allow_rotation=True'
        raise RuntimeError(msg)

    def apply_transformation(self, structure):
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Supercell Structure.
        """
        return structure * self.scaling_matrix

    def __repr__(self):
        return f'Supercell Transformation with scaling matrix {self.scaling_matrix}'

    @property
    def inverse(self):
        """Raises: NotImplementedError."""
        raise NotImplementedError

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False