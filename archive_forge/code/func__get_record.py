import os
from itertools import islice
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequentialAlignmentWriter
def _get_record(self, offset):
    """Retrieve a single MAF record located at the offset provided (PRIVATE)."""
    self._maf_fp.seek(offset)
    return next(self._mafiter)