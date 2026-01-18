import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _get_strand(self, seq_type):
    assert seq_type in ('hit', 'query')
    strand = getattr(self, '_%s_strand' % seq_type)
    if strand is None:
        frame = getattr(self, '%s_frame' % seq_type)
        if frame is not None:
            try:
                strand = frame // abs(frame)
            except ZeroDivisionError:
                strand = 0
            setattr(self, '%s_strand' % seq_type, strand)
    return strand