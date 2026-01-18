import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def _chunk_len_type(self):
    """
        Reads just enough of the input to
        determine the next chunk's length and type;
        return a (*length*, *type*) pair where *type* is a byte sequence.
        If there are no more chunks, ``None`` is returned.
        """
    x = self.file.read(8)
    if not x:
        return None
    if len(x) != 8:
        raise FormatError('End of file whilst reading chunk length and type.')
    length, type = struct.unpack('!I4s', x)
    if length > 2 ** 31 - 1:
        raise FormatError(f'Chunk {type} is too large: {length}.')
    type_bytes = set(bytearray(type))
    if not type_bytes <= set(range(65, 91)) | set(range(97, 123)):
        raise FormatError(f'Chunk {list(type)} has invalid Chunk Type.')
    return (length, type)