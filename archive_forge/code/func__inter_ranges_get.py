import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _inter_ranges_get(self, seq_type):
    assert seq_type in ('query', 'hit')
    strand = getattr(self, '%s_strand_all' % seq_type)[0]
    coords = getattr(self, '%s_range_all' % seq_type)
    if strand == -1:
        startfunc, endfunc = (min, max)
    else:
        startfunc, endfunc = (max, min)
    inter_coords = []
    for idx, coord in enumerate(coords[:-1]):
        start = startfunc(coords[idx])
        end = endfunc(coords[idx + 1])
        inter_coords.append((min(start, end), max(start, end)))
    return inter_coords