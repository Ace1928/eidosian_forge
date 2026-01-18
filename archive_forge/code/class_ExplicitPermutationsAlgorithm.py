from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
class ExplicitPermutationsAlgorithm(AbstractChemenvAlgorithm):
    """
    Class representing the algorithm doing the explicit permutations for the calculation of
    the Continuous Symmetry Measure.
    """

    def __init__(self, permutations):
        """Initializes a separation plane for a given perfect coordination geometry.

        Args:
            permutations: Permutations used for this algorithm.
        """
        super().__init__(algorithm_type=EXPLICIT_PERMUTATIONS)
        self._permutations = permutations

    def __str__(self):
        return self.algorithm_type

    @property
    def permutations(self):
        """
        Return the permutations to be performed for this algorithm.

        Returns:
            list: Permutations to be performed.
        """
        return self._permutations

    @property
    def as_dict(self):
        """
        Returns:
            dict: JSON-serializable representation of this ExplicitPermutationsAlgorithm
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'permutations': self._permutations}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstruct ExplicitPermutationsAlgorithm from its JSON-serializable dict representation.
        """
        return cls(dct['permutations'])