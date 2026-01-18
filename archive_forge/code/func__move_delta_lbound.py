import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def _move_delta_lbound(d, bytes):
    """Move the delta by the given amount of bytes, reducing its size so that its
    right bound stays static
    :param bytes: amount of bytes to move, must be smaller than delta size
    :return: d"""
    if bytes == 0:
        return
    d.to += bytes
    d.so += bytes
    d.ts -= bytes
    if d.data is not None:
        d.data = d.data[bytes:]
    return d