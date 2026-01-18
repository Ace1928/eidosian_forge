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
def _write_contig(self, record):
    max_len = self.MAX_WIDTH - self.HEADER_WIDTH
    lines = self._split_contig(record, max_len)
    for text in lines:
        self._write_single_line('CO', text)