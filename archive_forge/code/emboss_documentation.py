from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
Emboss alignment iterator.

    For reading the (pairwise) alignments from EMBOSS tools in what they
    call the "pairs" and "simple" formats.
    