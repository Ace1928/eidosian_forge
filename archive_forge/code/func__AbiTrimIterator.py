import datetime
import struct
import sys
from os.path import basename
from typing import List
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
def _AbiTrimIterator(handle):
    """Return an iterator for the Abi file format that yields trimmed SeqRecord objects (PRIVATE)."""
    return AbiIterator(handle, trim=True)