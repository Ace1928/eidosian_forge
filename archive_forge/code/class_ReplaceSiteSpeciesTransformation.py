from __future__ import annotations
import itertools
import logging
import math
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import EwaldMinimizer, EwaldSummation
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.transformation_abc import AbstractTransformation
class ReplaceSiteSpeciesTransformation(AbstractTransformation):
    """This transformation substitutes certain sites with certain species."""

    def __init__(self, indices_species_map):
        """
        Args:
            indices_species_map: A dict containing the species mapping in
                int-string pairs. E.g., { 1:"Na"} or {2:"Mn2+"}. Multiple
                substitutions can be done. Overloaded to accept sp_and_occu
                dictionary. E.g. {1: {"Ge":0.75, "C":0.25} }, which
                substitutes a single species with multiple species to generate a
                disordered structure.
        """
        self.indices_species_map = indices_species_map

    def apply_transformation(self, structure: Structure):
        """Apply the transformation.

        Args:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.

        Returns:
            A copy of structure with sites replaced.
        """
        struct = structure.copy()
        for idx, sp in self.indices_species_map.items():
            struct[int(idx)] = sp
        return struct

    def __repr__(self):
        return 'ReplaceSiteSpeciesTransformation :' + ', '.join([f'{key}->{val}' + val for key, val in self.indices_species_map.items()])

    @property
    def inverse(self):
        """Returns None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns False."""
        return False