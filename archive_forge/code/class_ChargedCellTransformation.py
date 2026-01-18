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
class ChargedCellTransformation(AbstractTransformation):
    """The ChargedCellTransformation applies a charge to a structure (or defect
    object).
    """

    def __init__(self, charge=0):
        """
        Args:
            charge: A integer charge to apply to the structure.
                Defaults to zero. Has to be a single integer. e.g. 2.
        """
        self.charge = charge

    def apply_transformation(self, structure):
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Charged Structure.
        """
        struct = structure.copy()
        struct.set_charge(self.charge)
        return struct

    def __repr__(self):
        return f'Structure with charge {self.charge}'

    @property
    def inverse(self):
        """Raises: NotImplementedError."""
        raise NotImplementedError

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False