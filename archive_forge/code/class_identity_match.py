import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
class identity_match:
    """Create a match function for use in an alignment.

    match and mismatch are the scores to give when two residues are equal
    or unequal.  By default, match is 1 and mismatch is 0.
    """

    def __init__(self, match=1, mismatch=0):
        """Initialize the class."""
        self.match = match
        self.mismatch = mismatch

    def __call__(self, charA, charB):
        """Call a match function instance already created."""
        if charA == charB:
            return self.match
        return self.mismatch