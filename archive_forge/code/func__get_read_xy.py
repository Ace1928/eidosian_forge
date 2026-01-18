import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def _get_read_xy(read_name):
    """Extract coordinates from last 5 characters of read name (PRIVATE)."""
    number = _string_as_base_36(read_name[9:])
    return divmod(number, 4096)