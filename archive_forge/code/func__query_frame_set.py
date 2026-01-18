import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _query_frame_set(self, value):
    """Set query sequence reading frame (PRIVATE)."""
    self._query_frame = self._prep_frame(value)