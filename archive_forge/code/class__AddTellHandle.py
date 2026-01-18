import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class _AddTellHandle:
    """Wrapper for handles which do not support the tell method (PRIVATE).

    Intended for use with things like network handles where tell (and reverse
    seek) are not supported. The SFF file needs to track the current offset in
    order to deal with the index block.
    """

    def __init__(self, handle):
        self._handle = handle
        self._offset = 0

    def read(self, length):
        data = self._handle.read(length)
        self._offset += len(data)
        return data

    def tell(self):
        return self._offset

    def seek(self, offset):
        if offset < self._offset:
            raise RuntimeError("Can't seek backwards")
        self._handle.read(offset - self._offset)

    def close(self):
        return self._handle.close()