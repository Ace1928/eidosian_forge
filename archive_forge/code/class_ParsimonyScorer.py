import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class ParsimonyScorer(Scorer):
    """Parsimony scorer with a scoring matrix.

    This is a combination of Fitch algorithm and Sankoff algorithm.
    See ParsimonyTreeConstructor for usage.

    :Parameters:
        matrix : _Matrix
            scoring matrix used in parsimony score calculation.

    """

    def __init__(self, matrix=None):
        """Initialize the class."""
        if not matrix or isinstance(matrix, _Matrix):
            self.matrix = matrix
        else:
            raise TypeError('Must provide a _Matrix object.')

    def get_score(self, tree, alignment):
        """Calculate parsimony score using the Fitch algorithm.

        Calculate and return the parsimony score given a tree and the
        MSA using either the Fitch algorithm (without a penalty matrix)
        or the Sankoff algorithm (with a matrix).
        """
        if not tree.is_bifurcating():
            raise ValueError('The tree provided should be bifurcating.')
        if not tree.rooted:
            tree.root_at_midpoint()
        terms = tree.get_terminals()
        terms.sort(key=lambda term: term.name)
        alignment.sort()
        if isinstance(alignment, MultipleSeqAlignment):
            if not all((t.name == a.id for t, a in zip(terms, alignment))):
                raise ValueError('Taxon names of the input tree should be the same with the alignment.')
        elif not all((t.name == s.id for t, s in zip(terms, alignment.sequences))):
            raise ValueError('Taxon names of the input tree should be the same with the alignment.')
        score = 0
        for i in range(len(alignment[0])):
            score_i = 0
            column_i = alignment[:, i]
            if column_i == len(column_i) * column_i[0]:
                continue
            if not self.matrix:
                clade_states = dict(zip(terms, [{c} for c in column_i]))
                for clade in tree.get_nonterminals(order='postorder'):
                    clade_childs = clade.clades
                    left_state = clade_states[clade_childs[0]]
                    right_state = clade_states[clade_childs[1]]
                    state = left_state & right_state
                    if not state:
                        state = left_state | right_state
                        score_i += 1
                    clade_states[clade] = state
            else:
                inf = float('inf')
                alphabet = self.matrix.names
                length = len(alphabet)
                clade_scores = {}
                for j in range(len(column_i)):
                    array = [inf] * length
                    index = alphabet.index(column_i[j])
                    array[index] = 0
                    clade_scores[terms[j]] = array
                for clade in tree.get_nonterminals(order='postorder'):
                    clade_childs = clade.clades
                    left_score = clade_scores[clade_childs[0]]
                    right_score = clade_scores[clade_childs[1]]
                    array = []
                    for m in range(length):
                        min_l = inf
                        min_r = inf
                        for n in range(length):
                            sl = self.matrix[alphabet[m], alphabet[n]] + left_score[n]
                            sr = self.matrix[alphabet[m], alphabet[n]] + right_score[n]
                            if min_l > sl:
                                min_l = sl
                            if min_r > sr:
                                min_r = sr
                        array.append(min_l + min_r)
                    clade_scores[clade] = array
                score_i = min(array)
            score += score_i
        return score