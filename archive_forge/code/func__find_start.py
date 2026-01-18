import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _find_start(score_matrix, best_score, align_globally):
    """Return a list of starting points (score, (row, col)) (PRIVATE).

    Indicating every possible place to start the tracebacks.
    """
    nrows, ncols = (len(score_matrix), len(score_matrix[0]))
    if align_globally:
        starts = [(best_score, (nrows - 1, ncols - 1))]
    else:
        starts = []
        tolerance = 0
        for row in range(nrows):
            for col in range(ncols):
                score = score_matrix[row][col]
                if rint(abs(score - best_score)) <= rint(tolerance):
                    starts.append((score, (row, col)))
    return starts