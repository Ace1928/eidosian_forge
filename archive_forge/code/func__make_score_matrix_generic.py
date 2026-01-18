import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _make_score_matrix_generic(sequenceA, sequenceB, match_fn, gap_A_fn, gap_B_fn, penalize_end_gaps, align_globally, score_only):
    """Generate a score and traceback matrix (PRIVATE).

    This implementation according to Needleman-Wunsch allows the usage of
    general gap functions and is rather slow. It is automatically called if
    you define your own gap functions. You can force the usage of this method
    with ``force_generic=True``.
    """
    local_max_score = 0
    lenA, lenB = (len(sequenceA), len(sequenceB))
    score_matrix, trace_matrix = ([], [])
    for i in range(lenA + 1):
        score_matrix.append([None] * (lenB + 1))
        if not score_only:
            trace_matrix.append([None] * (lenB + 1))
    for i in range(lenA + 1):
        if penalize_end_gaps[1]:
            score = gap_B_fn(0, i)
        else:
            score = 0.0
        score_matrix[i][0] = score
    for i in range(lenB + 1):
        if penalize_end_gaps[0]:
            score = gap_A_fn(0, i)
        else:
            score = 0.0
        score_matrix[0][i] = score
    for row in range(1, lenA + 1):
        for col in range(1, lenB + 1):
            nogap_score = score_matrix[row - 1][col - 1] + match_fn(sequenceA[row - 1], sequenceB[col - 1])
            if not penalize_end_gaps[0] and row == lenA:
                row_open = score_matrix[row][col - 1]
                row_extend = max((score_matrix[row][x] for x in range(col)))
            else:
                row_open = score_matrix[row][col - 1] + gap_A_fn(row, 1)
                row_extend = max((score_matrix[row][x] + gap_A_fn(row, col - x) for x in range(col)))
            if not penalize_end_gaps[1] and col == lenB:
                col_open = score_matrix[row - 1][col]
                col_extend = max((score_matrix[x][col] for x in range(row)))
            else:
                col_open = score_matrix[row - 1][col] + gap_B_fn(col, 1)
                col_extend = max((score_matrix[x][col] + gap_B_fn(col, row - x) for x in range(row)))
            best_score = max(nogap_score, row_open, row_extend, col_open, col_extend)
            local_max_score = max(local_max_score, best_score)
            if not align_globally and best_score < 0:
                score_matrix[row][col] = 0.0
            else:
                score_matrix[row][col] = best_score
            if not score_only:
                trace_score = 0
                if rint(nogap_score) == rint(best_score):
                    trace_score += 2
                if rint(row_open) == rint(best_score):
                    trace_score += 1
                if rint(row_extend) == rint(best_score):
                    trace_score += 8
                if rint(col_open) == rint(best_score):
                    trace_score += 4
                if rint(col_extend) == rint(best_score):
                    trace_score += 16
                trace_matrix[row][col] = trace_score
    if not align_globally:
        best_score = local_max_score
    return (score_matrix, trace_matrix, best_score)