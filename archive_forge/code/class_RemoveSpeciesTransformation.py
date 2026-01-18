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
class RemoveSpeciesTransformation(AbstractTransformation):
    """Remove all occurrences of some species from a structure."""

    def __init__(self, species_to_remove):
        """
        Args:
            species_to_remove: List of species to remove. E.g., ["Li", "Mn"].
        """
        self.species_to_remove = species_to_remove

    def apply_transformation(self, structure):
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Structure with species removed.
        """
        struct = structure.copy()
        for sp in self.species_to_remove:
            struct.remove_species([get_el_sp(sp)])
        return struct

    def __repr__(self):
        return 'Remove Species Transformation :' + ', '.join(self.species_to_remove)

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False