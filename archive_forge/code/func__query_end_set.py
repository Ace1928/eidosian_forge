import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _query_end_set(self, value):
    """Set the query sequence end coordinate (PRIVATE)."""
    self._query_end = self._prep_coord(value, 'query_start', ge)