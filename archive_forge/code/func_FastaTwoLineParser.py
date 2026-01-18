import warnings
from typing import Callable, Optional, Tuple
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
def FastaTwoLineParser(handle):
    """Iterate over no-wrapping Fasta records as string tuples.

    Arguments:
     - handle - input stream opened in text mode

    Functionally the same as SimpleFastaParser but with a strict
    interpretation of the FASTA format as exactly two lines per
    record, the greater-than-sign identifier with description,
    and the sequence with no line wrapping.

    Any line wrapping will raise an exception, as will excess blank
    lines (other than the special case of a zero-length sequence
    as the second line of a record).

    Examples
    --------
    This file uses two lines per FASTA record:

    >>> with open("Fasta/aster_no_wrap.pro") as handle:
    ...     for title, seq in FastaTwoLineParser(handle):
    ...         print("%s = %s..." % (title, seq[:3]))
    ...
    gi|3298468|dbj|BAA31520.1| SAMIPF = GGH...

    This equivalent file uses line wrapping:

    >>> with open("Fasta/aster.pro") as handle:
    ...     for title, seq in FastaTwoLineParser(handle):
    ...         print("%s = %s..." % (title, seq[:3]))
    ...
    Traceback (most recent call last):
       ...
    ValueError: Expected FASTA record starting with '>' character. Perhaps this file is using FASTA line wrapping? Got: 'MTFGLVYTVYATAIDPKKGSLGTIAPIAIGFIVGANI'

    """
    idx = -1
    for idx, line in enumerate(handle):
        if idx % 2 == 0:
            if line[0] != '>':
                raise ValueError(f"Expected FASTA record starting with '>' character. Perhaps this file is using FASTA line wrapping? Got: '{line}'")
            title = line[1:].rstrip()
        else:
            if line[0] == '>':
                raise ValueError(f"Two '>' FASTA lines in a row. Missing sequence line if this is strict two-line-per-record FASTA format. Have '>{title}' and '{line}'")
            yield (title, line.strip())
    if idx == -1:
        pass
    elif idx % 2 == 0:
        raise ValueError(f"Missing sequence line at end of file if this is strict two-line-per-record FASTA format. Have title line '{line}'")
    else:
        assert line[0] != '>', "line[0] == '>' ; this should be impossible!"