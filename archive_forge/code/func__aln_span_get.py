import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _aln_span_get(self):
    try:
        self._aln_span
    except AttributeError:
        if self.query is not None:
            self._aln_span = len(self.query)
        elif self.hit is not None:
            self._aln_span = len(self.hit)
    return self._aln_span