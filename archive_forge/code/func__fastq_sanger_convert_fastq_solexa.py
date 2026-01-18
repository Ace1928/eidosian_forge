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
def _fastq_sanger_convert_fastq_solexa(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    """Fast Sanger FASTQ to Solexa FASTQ conversion (PRIVATE).

    Avoids creating SeqRecord and Seq objects in order to speed up this
    conversion. Will issue a warning if the scores had to be truncated at 62
    (maximum possible in the Solexa FASTQ format)
    """
    trunc_char = chr(1)
    mapping = ''.join([chr(0) for ascii in range(33)] + [chr(64 + int(round(solexa_quality_from_phred(q)))) for q in range(62 + 1)] + [trunc_char for ascii in range(96, 127)] + [chr(0) for ascii in range(127, 256)])
    assert len(mapping) == 256
    return _fastq_generic2(in_file, out_file, mapping, trunc_char, 'Data loss - max Solexa quality 62 in Solexa FASTQ')