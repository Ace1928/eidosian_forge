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
class AddSitePropertyTransformation(AbstractTransformation):
    """Simple transformation to add site properties to a given structure."""

    def __init__(self, site_properties):
        """
        Args:
            site_properties (dict): site properties to be added to a structure.
        """
        self.site_properties = site_properties

    def apply_transformation(self, structure: Structure):
        """Apply the transformation.

        Args:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.

        Returns:
            A copy of structure with sites properties added.
        """
        new_struct = structure.copy()
        for prop in self.site_properties:
            new_struct.add_site_property(prop, self.site_properties[prop])
        return new_struct

    @property
    def inverse(self):
        """Returns None."""
        return

    @property
    def is_one_to_many(self) -> bool:
        """Returns False."""
        return False