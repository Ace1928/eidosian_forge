import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _hit_start_set(self, value):
    """Set the sequence hit start coordinate (PRIVATE)."""
    self._hit_start = self._prep_coord(value, 'hit_end', le)