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
class RemoveSitesTransformation(AbstractTransformation):
    """Remove certain sites in a structure."""

    def __init__(self, indices_to_remove):
        """
        Args:
            indices_to_remove: List of indices to remove. E.g., [0, 1, 2].
        """
        self.indices_to_remove = indices_to_remove

    def apply_transformation(self, structure: Structure):
        """Apply the transformation.

        Args:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.

        Returns:
            A copy of structure with sites removed.
        """
        struct = structure.copy()
        struct.remove_sites(self.indices_to_remove)
        return struct

    def __repr__(self):
        return 'RemoveSitesTransformation :' + ', '.join(map(str, self.indices_to_remove))

    @property
    def inverse(self):
        """Returns None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns False."""
        return False