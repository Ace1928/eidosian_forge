import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _find_gap_open(sequenceA, sequenceB, ali_seqA, ali_seqB, end, row, col, col_gap, gap_char, score_matrix, trace_matrix, in_process, gap_fn, target, index, direction, best_score, align_globally):
    """Find the starting point(s) of the extended gap (PRIVATE)."""
    dead_end = False
    target_score = score_matrix[row][col]
    for n in range(target):
        if direction == 'col':
            col -= 1
            ali_seqA += gap_char
            ali_seqB += sequenceB[col:col + 1]
        else:
            row -= 1
            ali_seqA += sequenceA[row:row + 1]
            ali_seqB += gap_char
        actual_score = score_matrix[row][col] + gap_fn(index, n + 1)
        if not align_globally and score_matrix[row][col] == best_score:
            dead_end = True
            break
        if rint(actual_score) == rint(target_score) and n > 0:
            if not trace_matrix[row][col]:
                break
            else:
                in_process.append((ali_seqA[:], ali_seqB[:], end, row, col, col_gap, trace_matrix[row][col]))
        if not trace_matrix[row][col]:
            dead_end = True
    return (ali_seqA, ali_seqB, row, col, in_process, dead_end)