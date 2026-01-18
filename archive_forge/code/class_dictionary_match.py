import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
class dictionary_match:
    """Create a match function for use in an alignment.

    Attributes:
     - score_dict     - A dictionary where the keys are tuples (residue 1,
       residue 2) and the values are the match scores between those residues.
     - symmetric      - A flag that indicates whether the scores are symmetric.

    """

    def __init__(self, score_dict, symmetric=1):
        """Initialize the class."""
        if isinstance(score_dict, substitution_matrices.Array):
            score_dict = dict(score_dict)
        self.score_dict = score_dict
        self.symmetric = symmetric

    def __call__(self, charA, charB):
        """Call a dictionary match instance already created."""
        if self.symmetric and (charA, charB) not in self.score_dict:
            charB, charA = (charA, charB)
        return self.score_dict[charA, charB]