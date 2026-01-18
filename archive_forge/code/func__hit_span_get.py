import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _hit_span_get(self):
    """Return the number of residues covered by the hit sequence (PRIVATE)."""
    try:
        return self.hit_end - self.hit_start
    except TypeError:
        return None