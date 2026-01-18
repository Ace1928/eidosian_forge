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
def _fastq_illumina_convert_qual(in_file: _TextIOSource, out_file: _TextIOSource) -> int:
    """Fast Illumina 1.3+ FASTQ to QUAL conversion (PRIVATE)."""
    mapping = {chr(q + 64): str(q) for q in range(62 + 1)}
    return _fastq_convert_qual(in_file, out_file, mapping)