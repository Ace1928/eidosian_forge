import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
def _nni(self, starting_tree, alignment):
    """Search for the best parsimony tree using the NNI algorithm (PRIVATE)."""
    best_tree = starting_tree
    while True:
        best_score = self.scorer.get_score(best_tree, alignment)
        temp = best_score
        for t in self._get_neighbors(best_tree):
            score = self.scorer.get_score(t, alignment)
            if score < best_score:
                best_score = score
                best_tree = t
        if best_score >= temp:
            break
    return best_tree