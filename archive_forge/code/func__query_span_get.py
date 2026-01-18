import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _query_span_get(self):
    """Return the number or residues covered by the query (PRIVATE)."""
    try:
        return self.query_end - self.query_start
    except TypeError:
        return None