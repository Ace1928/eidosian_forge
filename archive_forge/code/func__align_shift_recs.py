import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def _align_shift_recs(recs):
    """Build alignment according to the frameshift detected by _check_corr (PRIVATE).

    Argument:
     - recs - a list of SeqRecords containing a CodonSeq dictated
       by a rf_table (with frameshift in some of them).

    """

    def find_next_int(k, lst):
        idx = lst.index(k)
        p = 0
        while True:
            if isinstance(lst[idx + p], int):
                return (lst[idx + p], p)
            p += 1
    full_rf_table_lst = [rec.seq.get_full_rf_table() for rec in recs]
    rf_num = [0] * len(recs)
    for k, rec in enumerate(recs):
        for i in rec.seq.get_full_rf_table():
            if isinstance(i, int):
                rf_num[k] += 1
            elif rec.seq[int(i):int(i) + 3] == '---':
                rf_num[k] += 1
    if len(set(rf_num)) != 1:
        raise RuntimeError('Number of alignable codons unequal in given records')
    i = 0
    rec_num = len(recs)
    while True:
        add_lst = []
        try:
            col_rf_lst = [k[i] for k in full_rf_table_lst]
        except IndexError:
            break
        for j, k in enumerate(col_rf_lst):
            add_lst.append((j, int(k)))
            if isinstance(k, float) and recs[j].seq[int(k):int(k) + 3] != '---':
                m, p = find_next_int(k, full_rf_table_lst[j])
                if (m - k) % 3 != 0:
                    gap_num = 3 - (m - k) % 3
                else:
                    gap_num = 0
                if gap_num != 0:
                    gaps = '-' * int(gap_num)
                    seq = CodonSeq(rf_table=recs[j].seq.rf_table)
                    seq += recs[j].seq[:int(k)] + gaps + recs[j].seq[int(k):]
                    full_rf_table = full_rf_table_lst[j]
                    bp = full_rf_table.index(k)
                    full_rf_table = full_rf_table[:bp] + [v + int(gap_num) for v in full_rf_table[bp + 1:]]
                    full_rf_table_lst[j] = full_rf_table
                    recs[j].seq = seq
                add_lst.pop()
                gap_num += m - k
                i += p - 1
        if len(add_lst) != rec_num:
            for j, k in add_lst:
                seq = CodonSeq(rf_table=recs[j].seq.rf_table)
                gaps = '-' * int(gap_num)
                seq += recs[j].seq[:int(k)] + gaps + recs[j].seq[int(k):]
                full_rf_table = full_rf_table_lst[j]
                bp = full_rf_table.index(k)
                inter_rf = []
                for t in range(0, len(gaps), 3):
                    inter_rf.append(k + t + 3.0)
                full_rf_table = full_rf_table[:bp] + inter_rf + [v + int(gap_num) for v in full_rf_table[bp:]]
                full_rf_table_lst[j] = full_rf_table
                recs[j].seq = seq
        i += 1
    return recs