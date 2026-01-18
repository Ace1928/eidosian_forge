import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class _SffTrimIterator(SffIterator):
    """Iterate over SFF reads (as SeqRecord objects) with trimming (PRIVATE)."""

    def __init__(self, source):
        super().__init__(source, trim=True)