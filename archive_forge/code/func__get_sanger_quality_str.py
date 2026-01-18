import warnings
from math import log
from Bio import BiopythonParserWarning
from Bio import BiopythonWarning
from Bio import StreamModeError
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from typing import (
def _get_sanger_quality_str(record: SeqRecord) -> str:
    """Return a Sanger FASTQ encoded quality string (PRIVATE).

    >>> from Bio.Seq import Seq
    >>> from Bio.SeqRecord import SeqRecord
    >>> r = SeqRecord(Seq("ACGTAN"), id="Test",
    ...               letter_annotations = {"phred_quality":[50, 40, 30, 20, 10, 0]})
    >>> _get_sanger_quality_str(r)
    'SI?5+!'

    If as in the above example (or indeed a SeqRecord parser with Bio.SeqIO),
    the PHRED qualities are integers, this function is able to use a very fast
    pre-cached mapping. However, if they are floats which differ slightly, then
    it has to do the appropriate rounding - which is slower:

    >>> r2 = SeqRecord(Seq("ACGTAN"), id="Test2",
    ...      letter_annotations = {"phred_quality":[50.0, 40.05, 29.99, 20, 9.55, 0.01]})
    >>> _get_sanger_quality_str(r2)
    'SI?5+!'

    If your scores include a None value, this raises an exception:

    >>> r3 = SeqRecord(Seq("ACGTAN"), id="Test3",
    ...               letter_annotations = {"phred_quality":[50, 40, 30, 20, 10, None]})
    >>> _get_sanger_quality_str(r3)
    Traceback (most recent call last):
       ...
    TypeError: A quality value of None was found

    If (strangely) your record has both PHRED and Solexa scores, then the PHRED
    scores are used in preference:

    >>> r4 = SeqRecord(Seq("ACGTAN"), id="Test4",
    ...               letter_annotations = {"phred_quality":[50, 40, 30, 20, 10, 0],
    ...                                     "solexa_quality":[-5, -4, 0, None, 0, 40]})
    >>> _get_sanger_quality_str(r4)
    'SI?5+!'

    If there are no PHRED scores, but there are Solexa scores, these are used
    instead (after the appropriate conversion):

    >>> r5 = SeqRecord(Seq("ACGTAN"), id="Test5",
    ...      letter_annotations = {"solexa_quality":[40, 30, 20, 10, 0, -5]})
    >>> _get_sanger_quality_str(r5)
    'I?5+$"'

    Again, integer Solexa scores can be looked up in a pre-cached mapping making
    this very fast. You can still use approximate floating point scores:

    >>> r6 = SeqRecord(Seq("ACGTAN"), id="Test6",
    ...      letter_annotations = {"solexa_quality":[40.1, 29.7, 20.01, 10, 0.0, -4.9]})
    >>> _get_sanger_quality_str(r6)
    'I?5+$"'

    Notice that due to the limited range of printable ASCII characters, a
    PHRED quality of 93 is the maximum that can be held in an Illumina FASTQ
    file (using ASCII 126, the tilde). This function will issue a warning
    in this situation.
    """
    try:
        qualities = record.letter_annotations['phred_quality']
    except KeyError:
        pass
    else:
        try:
            return ''.join((_phred_to_sanger_quality_str[qp] for qp in qualities))
        except KeyError:
            pass
        if None in qualities:
            raise TypeError('A quality value of None was found')
        if max(qualities) >= 93.5:
            warnings.warn('Data loss - max PHRED quality 93 in Sanger FASTQ', BiopythonWarning)
        return ''.join((chr(min(126, int(round(qp)) + SANGER_SCORE_OFFSET)) for qp in qualities))
    try:
        qualities = record.letter_annotations['solexa_quality']
    except KeyError:
        raise ValueError('No suitable quality scores found in letter_annotations of SeqRecord (id=%s).' % record.id) from None
    try:
        return ''.join((_solexa_to_sanger_quality_str[qs] for qs in qualities))
    except KeyError:
        pass
    if None in qualities:
        raise TypeError('A quality value of None was found')
    if max(qualities) >= 93.5:
        warnings.warn('Data loss - max PHRED quality 93 in Sanger FASTQ', BiopythonWarning)
    return ''.join((chr(min(126, int(round(phred_quality_from_solexa(qs))) + SANGER_SCORE_OFFSET)) for qs in qualities))