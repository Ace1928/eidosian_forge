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
class TranslateSitesTransformation(AbstractTransformation):
    """This class translates a set of sites by a certain vector."""

    def __init__(self, indices_to_move, translation_vector, vector_in_frac_coords=True):
        """
        Args:
            indices_to_move: The indices of the sites to move
            translation_vector: Vector to move the sites. If a list of list or numpy
                array of shape, (len(indices_to_move), 3), is provided then each
                translation vector is applied to the corresponding site in the
                indices_to_move.
            vector_in_frac_coords: Set to True if the translation vector is in
                fractional coordinates, and False if it is in cartesian
                coordinations. Defaults to True.
        """
        self.indices_to_move = indices_to_move
        self.translation_vector = np.array(translation_vector)
        self.vector_in_frac_coords = vector_in_frac_coords

    def apply_transformation(self, structure: Structure):
        """Apply the transformation.

        Args:
            structure (Structure): A structurally similar structure in
                regards to crystal and site positions.

        Returns:
            A copy of structure with sites translated.
        """
        struct = structure.copy()
        if self.translation_vector.shape == (len(self.indices_to_move), 3):
            for i, idx in enumerate(self.indices_to_move):
                struct.translate_sites(idx, self.translation_vector[i], self.vector_in_frac_coords)
        else:
            struct.translate_sites(self.indices_to_move, self.translation_vector, self.vector_in_frac_coords)
        return struct

    def __repr__(self):
        return f'TranslateSitesTransformation for indices {self.indices_to_move}, vect {self.translation_vector} and vect_in_frac_coords = {self.vector_in_frac_coords}'

    @property
    def inverse(self):
        """
        Returns:
            TranslateSitesTransformation with the reverse translation.
        """
        return TranslateSitesTransformation(self.indices_to_move, -self.translation_vector, self.vector_in_frac_coords)

    @property
    def is_one_to_many(self) -> bool:
        """Returns False."""
        return False

    def as_dict(self):
        """JSON-serializable dict representation."""
        dct = MSONable.as_dict(self)
        dct['translation_vector'] = self.translation_vector.tolist()
        return dct