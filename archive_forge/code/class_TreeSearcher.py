import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class TreeSearcher:
    """Base class for all tree searching methods."""

    def search(self, starting_tree, alignment):
        """Caller to search the best tree with a starting tree.

        This should be implemented in subclass.
        """
        raise NotImplementedError('Method not implemented!')