from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def _get_codon_list(codonseq):
    """List of codons according to full_rf_table for counting (PRIVATE)."""
    full_rf_table = codonseq.get_full_rf_table()
    codon_lst = []
    for i, k in enumerate(full_rf_table):
        if isinstance(k, int):
            start = k
            try:
                end = int(full_rf_table[i + 1])
            except IndexError:
                end = start + 3
            this_codon = str(codonseq[start:end])
            if len(this_codon) == 3:
                codon_lst.append(this_codon)
            else:
                codon_lst.append(str(this_codon.ungap()))
        elif str(codonseq[int(k):int(k) + 3]) == '---':
            codon_lst.append('---')
        else:
            codon_lst.append(codonseq[int(k):int(k) + 3])
    return codon_lst