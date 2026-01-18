import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _prep_frame(self, frame):
    if frame not in (-3, -2, -1, 0, 1, 2, 3, None):
        raise ValueError('Strand should be an integer between -3 and 3, or None; not %r' % frame)
    return frame