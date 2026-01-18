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
def _get_illumina_quality_str(record: SeqRecord) -> str:
    """Return an Illumina 1.3 to 1.7 FASTQ encoded quality string (PRIVATE).

    Notice that due to the limited range of printable ASCII characters, a
    PHRED quality of 62 is the maximum that can be held in an Illumina FASTQ
    file (using ASCII 126, the tilde). This function will issue a warning
    in this situation.
    """
    try:
        qualities = record.letter_annotations['phred_quality']
    except KeyError:
        pass
    else:
        try:
            return ''.join((_phred_to_illumina_quality_str[qp] for qp in qualities))
        except KeyError:
            pass
        if None in qualities:
            raise TypeError('A quality value of None was found')
        if max(qualities) >= 62.5:
            warnings.warn('Data loss - max PHRED quality 62 in Illumina FASTQ', BiopythonWarning)
        return ''.join((chr(min(126, int(round(qp)) + SOLEXA_SCORE_OFFSET)) for qp in qualities))
    try:
        qualities = record.letter_annotations['solexa_quality']
    except KeyError:
        raise ValueError('No suitable quality scores found in letter_annotations of SeqRecord (id=%s).' % record.id) from None
    try:
        return ''.join((_solexa_to_illumina_quality_str[qs] for qs in qualities))
    except KeyError:
        pass
    if None in qualities:
        raise TypeError('A quality value of None was found')
    if max(qualities) >= 62.5:
        warnings.warn('Data loss - max PHRED quality 62 in Illumina FASTQ', BiopythonWarning)
    return ''.join((chr(min(126, int(round(phred_quality_from_solexa(qs))) + SOLEXA_SCORE_OFFSET)) for qs in qualities))