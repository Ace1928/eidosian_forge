from Bio.Seq import Seq
import re
import math
from Bio import motifs
from Bio import Align
def _read_sites(handle):
    """Read the motif from JASPAR .sites file (PRIVATE)."""
    alphabet = 'ACGT'
    instances = []
    for line in handle:
        if not line.startswith('>'):
            break
        line = next(handle)
        instance = ''
        for c in line.strip():
            if c.isupper():
                instance += c
        instance = Seq(instance)
        instances.append(instance)
    alignment = Align.Alignment(instances)
    motif = Motif(matrix_id=None, name=None, alphabet=alphabet, alignment=alignment)
    motif.mask = '*' * motif.length
    record = Record()
    record.append(motif)
    return record