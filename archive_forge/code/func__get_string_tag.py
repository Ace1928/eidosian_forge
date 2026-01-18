import datetime
import struct
import sys
from os.path import basename
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _get_string_tag(opt_bytes_value, default=None):
    """Return the string value of the given an optional raw bytes tag value.

    If the bytes value is None, return the given default value.

    """
    if opt_bytes_value is None:
        return default
    try:
        return opt_bytes_value.decode()
    except UnicodeDecodeError:
        return opt_bytes_value.decode(encoding=sys.getdefaultencoding())