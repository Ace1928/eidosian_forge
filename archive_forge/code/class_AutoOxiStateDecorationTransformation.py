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
class AutoOxiStateDecorationTransformation(AbstractTransformation):
    """This transformation automatically decorates a structure with oxidation
    states using a bond valence approach.
    """

    def __init__(self, symm_tol=0.1, max_radius=4, max_permutations=100000, distance_scale_factor=1.015):
        """
        Args:
            symm_tol (float): Symmetry tolerance used to determine which sites are
                symmetrically equivalent. Set to 0 to turn off symmetry.
            max_radius (float): Maximum radius in Angstrom used to find nearest
                neighbors.
            max_permutations (int): Maximum number of permutations of oxidation
                states to test.
            distance_scale_factor (float): A scale factor to be applied. This is
                useful for scaling distances, esp in the case of
                calculation-relaxed structures, which may tend to under (GGA) or
                over bind (LDA). The default of 1.015 works for GGA. For
                experimental structure, set this to 1.
        """
        self.symm_tol = symm_tol
        self.max_radius = max_radius
        self.max_permutations = max_permutations
        self.distance_scale_factor = distance_scale_factor
        self.analyzer = BVAnalyzer(symm_tol, max_radius, max_permutations, distance_scale_factor)

    def apply_transformation(self, structure):
        """Apply the transformation.

        Args:
            structure (Structure): Input Structure

        Returns:
            Oxidation state decorated Structure.
        """
        return self.analyzer.get_oxi_state_decorated_structure(structure)

    @property
    def inverse(self):
        """Returns: None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns: False."""
        return False