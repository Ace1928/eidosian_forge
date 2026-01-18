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
def _write_single_line(self, tag, text):
    assert len(tag) == 2
    line = tag + '   ' + text
    if len(text) > self.MAX_WIDTH:
        warnings.warn(f'Line {line!r} too long', BiopythonWarning)
    self.handle.write(line + '\n')