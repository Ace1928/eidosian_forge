import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
@staticmethod
def _get_annotation_str(record, key, default='.', just_first=False):
    """Get an annotation dictionary entry (as a string) (PRIVATE).

        Some entries are lists, in which case if just_first=True the first entry
        is returned.  If just_first=False (default) this verifies there is only
        one entry before returning it.
        """
    try:
        answer = record.annotations[key]
    except KeyError:
        return default
    if isinstance(answer, list):
        if not just_first:
            assert len(answer) == 1
        return str(answer[0])
    else:
        return str(answer)