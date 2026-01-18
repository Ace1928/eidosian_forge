import warnings
from typing import Callable, Optional, Tuple
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
def SimpleFastaParser(handle):
    """Iterate over Fasta records as string tuples.

    Arguments:
     - handle - input stream opened in text mode

    For each record a tuple of two strings is returned, the FASTA title
    line (without the leading '>' character), and the sequence (with any
    whitespace removed). The title line is not divided up into an
    identifier (the first word) and comment or description.

    >>> with open("Fasta/dups.fasta") as handle:
    ...     for values in SimpleFastaParser(handle):
    ...         print(values)
    ...
    ('alpha', 'ACGTA')
    ('beta', 'CGTC')
    ('gamma', 'CCGCC')
    ('alpha (again - this is a duplicate entry to test the indexing code)', 'ACGTA')
    ('delta', 'CGCGC')

    """
    for line in handle:
        if line[0] == '>':
            title = line[1:].rstrip()
            break
    else:
        return
    lines = []
    for line in handle:
        if line[0] == '>':
            yield (title, ''.join(lines).replace(' ', '').replace('\r', ''))
            lines = []
            title = line[1:].rstrip()
            continue
        lines.append(line.rstrip())
    yield (title, ''.join(lines).replace(' ', '').replace('\r', ''))