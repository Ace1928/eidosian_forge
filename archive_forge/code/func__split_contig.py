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
def _split_contig(self, record, max_len):
    """Return a list of strings, splits on commas (PRIVATE)."""
    contig = record.annotations.get('contig', '')
    if isinstance(contig, (list, tuple)):
        contig = ''.join(contig)
    contig = self.clean(contig)
    answer = []
    while contig:
        if len(contig) > max_len:
            pos = contig[:max_len - 1].rfind(',')
            if pos == -1:
                raise ValueError('Could not break up CONTIG')
            text, contig = (contig[:pos + 1], contig[pos + 1:])
        else:
            text, contig = (contig, '')
        answer.append(text)
    return answer