import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
def GC123(seq):
    """Calculate G+C content: total, for first, second and third positions.

    Returns a tuple of four floats (percentages between 0 and 100) for the
    entire sequence, and the three codon positions.  e.g.

    >>> from Bio.SeqUtils import GC123
    >>> GC123("ACTGTN")
    (40.0, 50.0, 50.0, 0.0)

    Copes with mixed case sequences, but does NOT deal with ambiguous
    nucleotides.
    """
    d = {}
    for nt in ['A', 'T', 'G', 'C']:
        d[nt] = [0, 0, 0]
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        if len(codon) < 3:
            codon += '  '
        for pos in range(3):
            for nt in ['A', 'T', 'G', 'C']:
                if codon[pos] == nt or codon[pos] == nt.lower():
                    d[nt][pos] += 1
    gc = {}
    gcall = 0
    nall = 0
    for i in range(3):
        try:
            n = d['G'][i] + d['C'][i] + d['T'][i] + d['A'][i]
            gc[i] = (d['G'][i] + d['C'][i]) * 100.0 / n
        except Exception:
            gc[i] = 0
        gcall = gcall + d['G'][i] + d['C'][i]
        nall = nall + n
    gcall = 100.0 * gcall / nall
    return (gcall, gc[0], gc[1], gc[2])