import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
def _height_of(self, clade):
    """Calculate clade height -- the longest path to any terminal (PRIVATE)."""
    height = 0
    if clade.is_terminal():
        height = clade.branch_length
    else:
        height = height + max((self._height_of(c) for c in clade.clades))
    return height