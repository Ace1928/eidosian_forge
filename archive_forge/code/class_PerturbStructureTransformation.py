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
class PerturbStructureTransformation(AbstractTransformation):
    """This transformation perturbs a structure by a specified distance in random
    directions. Used for breaking symmetries.
    """

    def __init__(self, distance: float=0.01, min_distance: float | None=None):
        """
        Args:
            distance: Distance of perturbation in angstroms. All sites
                will be perturbed by exactly that distance in a random
                direction.
            min_distance: if None, all displacements will be equidistant. If int
                or float, perturb each site a distance drawn from the uniform
                distribution between 'min_distance' and 'distance'.
        """
        self.distance = distance
        self.min_distance = min_distance

    def apply_transformation(self, structure: Structure) -> Structure:
        """Apply the transformation.

        Args:
            structure: Input Structure

        Returns:
            Structure with sites perturbed.
        """
        struct = structure.copy()
        struct.perturb(self.distance, min_distance=self.min_distance)
        return struct

    def __repr__(self):
        return f'PerturbStructureTransformation : Min_distance = {self.min_distance}'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False