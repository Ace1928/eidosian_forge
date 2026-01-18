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
class PrimitiveCellTransformation(AbstractTransformation):
    """This class finds the primitive cell of the input structure.
    It returns a structure that is not necessarily orthogonalized
    Author: Will Richards.
    """

    def __init__(self, tolerance=0.5):
        """
        Args:
            tolerance (float): Tolerance for each coordinate of a particular
                site. For example, [0.5, 0, 0.5] in Cartesian coordinates will be
                considered to be on the same coordinates as [0, 0, 0] for a
                tolerance of 0.5. Defaults to 0.5.
        """
        self.tolerance = tolerance

    def apply_transformation(self, structure):
        """Returns most primitive cell for structure.

        Args:
            structure: A structure

        Returns:
            The most primitive structure found. The returned structure is
            guaranteed to have len(new structure) <= len(structure).
        """
        return structure.get_primitive_structure(tolerance=self.tolerance)

    def __repr__(self):
        return 'Primitive cell transformation'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False