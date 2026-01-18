from math import sqrt, erfc
import warnings
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio import BiopythonWarning
from Bio.codonalign.codonseq import _get_codon_list, CodonSeq, cal_dn_ds
Convert a MultipleSeqAlignment to CodonAlignment.

        Function to convert a MultipleSeqAlignment to CodonAlignment.
        It is the user's responsibility to ensure all the requirement
        needed by CodonAlignment is met.
        