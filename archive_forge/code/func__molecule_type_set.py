import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _molecule_type_set(self, value):
    self._molecule_type = value
    try:
        self.query.annotations['molecule_type'] = value
    except AttributeError:
        pass
    try:
        self.hit.annotations['molecule_type'] = value
    except AttributeError:
        pass