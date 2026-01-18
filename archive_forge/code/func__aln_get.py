import warnings
from operator import ge, le
from Bio import BiopythonWarning
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SearchIO._utils import (
from ._base import _BaseHSP
def _aln_get(self):
    if self.query is None and self.hit is None:
        return None
    if self.hit is None:
        msa = MultipleSeqAlignment([self.query])
    elif self.query is None:
        msa = MultipleSeqAlignment([self.hit])
    else:
        msa = MultipleSeqAlignment([self.query, self.hit])
    molecule_type = self.molecule_type
    if molecule_type is not None:
        msa.molecule_type = molecule_type
    return msa