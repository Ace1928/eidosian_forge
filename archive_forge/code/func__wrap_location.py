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
def _wrap_location(self, location):
    """Split a feature location into lines (break at commas) (PRIVATE)."""
    length = self.MAX_WIDTH - self.QUALIFIER_INDENT
    if len(location) <= length:
        return location
    index = location[:length].rfind(',')
    if index == -1:
        warnings.warn(f"Couldn't split location:\n{location}", BiopythonWarning)
        return location
    return location[:index + 1] + '\n' + self.QUALIFIER_INDENT_STR + self._wrap_location(location[index + 1:])