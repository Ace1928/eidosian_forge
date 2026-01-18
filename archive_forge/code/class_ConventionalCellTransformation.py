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
class ConventionalCellTransformation(AbstractTransformation):
    """This class finds the conventional cell of the input structure."""

    def __init__(self, symprec: float=0.01, angle_tolerance=5, international_monoclinic=True):
        """
        Args:
            symprec (float): tolerance as in SpacegroupAnalyzer
            angle_tolerance (float): angle tolerance as in SpacegroupAnalyzer
            international_monoclinic (bool): whether to use beta (True) or alpha (False)
        as the non-right-angle in the unit cell.
        """
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.international_monoclinic = international_monoclinic

    def apply_transformation(self, structure):
        """Returns most primitive cell for structure.

        Args:
            structure: A structure

        Returns:
            The same structure in a conventional standard setting
        """
        sga = SpacegroupAnalyzer(structure, symprec=self.symprec, angle_tolerance=self.angle_tolerance)
        return sga.get_conventional_standard_structure(international_monoclinic=self.international_monoclinic)

    def __repr__(self):
        return 'Conventional cell transformation'

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False