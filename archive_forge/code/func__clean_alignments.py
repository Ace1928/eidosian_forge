import warnings
from collections import namedtuple
from Bio import BiopythonWarning
from Bio import BiopythonDeprecationWarning
from Bio.Align import substitution_matrices
def _clean_alignments(alignments):
    """Take a list of alignments and return a cleaned version (PRIVATE).

    Remove duplicates, make sure begin and end are set correctly, remove
    empty alignments.
    """
    unique_alignments = []
    for align in alignments:
        if align not in unique_alignments:
            unique_alignments.append(align)
    i = 0
    while i < len(unique_alignments):
        seqA, seqB, score, begin, end = unique_alignments[i]
        if end is None:
            end = len(seqA)
        elif end < 0:
            end = end + len(seqA)
        if begin >= end:
            del unique_alignments[i]
            continue
        unique_alignments[i] = Alignment(seqA, seqB, score, begin, end)
        i += 1
    return unique_alignments