import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
def GC_skew(seq, window=100):
    """Calculate GC skew (G-C)/(G+C) for multiple windows along the sequence.

    Returns a list of ratios (floats), controlled by the length of the sequence
    and the size of the window.

    Returns 0 for windows without any G/C by handling zero division errors.

    Does NOT look at any ambiguous nucleotides.
    """
    values = []
    for i in range(0, len(seq), window):
        s = seq[i:i + window]
        g = s.count('G') + s.count('g')
        c = s.count('C') + s.count('c')
        try:
            skew = (g - c) / (g + c)
        except ZeroDivisionError:
            skew = 0.0
        values.append(skew)
    return values