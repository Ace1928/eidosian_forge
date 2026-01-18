import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _finish_backtrace(sequenceA, sequenceB, ali_seqA, ali_seqB, row, col, gap_char):
    """Add remaining sequences and fill with gaps if necessary (PRIVATE)."""
    if row:
        ali_seqA += sequenceA[row - 1::-1]
    if col:
        ali_seqB += sequenceB[col - 1::-1]
    if row > col:
        ali_seqB += gap_char * (len(ali_seqA) - len(ali_seqB))
    elif col > row:
        ali_seqA += gap_char * (len(ali_seqB) - len(ali_seqA))
    return (ali_seqA, ali_seqB)