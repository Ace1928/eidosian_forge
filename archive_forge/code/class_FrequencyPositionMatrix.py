import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
class FrequencyPositionMatrix(GenericPositionMatrix):
    """Class for the support of frequency calculations on the Position Matrix."""

    def normalize(self, pseudocounts=None):
        """Create and return a position-weight matrix by normalizing the counts matrix.

        If pseudocounts is None (default), no pseudocounts are added
        to the counts.

        If pseudocounts is a number, it is added to the counts before
        calculating the position-weight matrix.

        Alternatively, the pseudocounts can be a dictionary with a key
        for each letter in the alphabet associated with the motif.
        """
        counts = {}
        if pseudocounts is None:
            for letter in self.alphabet:
                counts[letter] = [0.0] * self.length
        elif isinstance(pseudocounts, dict):
            for letter in self.alphabet:
                counts[letter] = [float(pseudocounts[letter])] * self.length
        else:
            for letter in self.alphabet:
                counts[letter] = [float(pseudocounts)] * self.length
        for i in range(self.length):
            for letter in self.alphabet:
                counts[letter][i] += self[letter][i]
        return PositionWeightMatrix(self.alphabet, counts)