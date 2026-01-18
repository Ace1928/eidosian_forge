import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _reverse_matrices(score_matrix, trace_matrix):
    """Reverse score and trace matrices (PRIVATE)."""
    reverse_score_matrix = []
    reverse_trace_matrix = []
    reverse_trace = {1: 4, 2: 2, 3: 6, 4: 1, 5: 5, 6: 3, 7: 7, 8: 16, 9: 20, 10: 18, 11: 22, 12: 17, 13: 21, 14: 19, 15: 23, 16: 8, 17: 12, 18: 10, 19: 14, 20: 9, 21: 13, 22: 11, 23: 15, 24: 24, 25: 28, 26: 26, 27: 30, 28: 25, 29: 29, 30: 27, 31: 31, None: None}
    for col in range(len(score_matrix[0])):
        new_score_row = []
        new_trace_row = []
        for row in range(len(score_matrix)):
            new_score_row.append(score_matrix[row][col])
            new_trace_row.append(reverse_trace[trace_matrix[row][col]])
        reverse_score_matrix.append(new_score_row)
        reverse_trace_matrix.append(new_trace_row)
    return (reverse_score_matrix, reverse_trace_matrix)