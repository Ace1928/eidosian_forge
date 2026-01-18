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
def _fastq_generic2(in_file: _TextIOSource, out_file: _TextIOSource, mapping: Union[Sequence[str], Mapping[int, Optional[Union[str, int]]]], truncate_char: str, truncate_msg: str) -> int:
    """FASTQ helper function where there could be data loss by truncation (PRIVATE)."""
    count = 0
    null = chr(0)
    with as_handle(out_file, 'w') as out_handle:
        for title, seq, old_qual in FastqGeneralIterator(in_file):
            count += 1
            qual = old_qual.translate(mapping)
            if null in qual:
                raise ValueError('Invalid character in quality string')
            if truncate_char in qual:
                qual = qual.replace(truncate_char, chr(126))
                warnings.warn(truncate_msg, BiopythonWarning)
            out_handle.write(f'@{title}\n{seq}\n+\n{qual}\n')
    return count