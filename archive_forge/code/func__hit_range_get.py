import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _hit_range_get(self):
    """Return the start and end of a hit (PRIVATE)."""
    return (self.hit_start, self.hit_end)