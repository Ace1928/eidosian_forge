import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def _get_aa_regex(codon_table, stop='*', unknown='X'):
    """Set up the regular expression of a given CodonTable (PRIVATE).

    >>> from Bio.Data.CodonTable import generic_by_id
    >>> p = generic_by_id[1]
    >>> t = _get_aa_regex(p)
    >>> print(t['A'][0])
    G
    >>> print(t['A'][1])
    C
    >>> print(sorted(list(t['A'][2:])))
    ['A', 'C', 'G', 'T', 'U', '[', ']']
    >>> print(sorted(list(t['L'][:5])))
    ['C', 'T', 'U', '[', ']']
    >>> print(sorted(list(t['L'][5:9])))
    ['T', 'U', '[', ']']
    >>> print(sorted(list(t['L'][9:])))
    ['A', 'C', 'G', 'T', 'U', '[', ']']

    """
    from Bio.Data.CodonTable import CodonTable
    if not isinstance(codon_table, CodonTable):
        raise TypeError('Input table is not a instance of Bio.Data.CodonTable object')
    aa2codon = {}
    for codon, aa in codon_table.forward_table.items():
        aa2codon.setdefault(aa, []).append(codon)
    for aa, codons in aa2codon.items():
        aa2codon[aa] = _codons2re(codons)
    aa2codon[stop] = _codons2re(codon_table.stop_codons)
    aa2codon[unknown] = '...'
    return aa2codon