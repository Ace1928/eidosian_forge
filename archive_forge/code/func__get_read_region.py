import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _get_read_region(read_name):
    """Extract region from read name (PRIVATE)."""
    return int(read_name[8])