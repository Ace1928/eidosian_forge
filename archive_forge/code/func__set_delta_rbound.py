import zlib
from gitdb.util import byte_ord
import mmap
from itertools import islice
from functools import reduce
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_text
from gitdb.typ import (
from io import StringIO
def _set_delta_rbound(d, size):
    """Truncate the given delta to the given size
    :param size: size relative to our target offset, may not be 0, must be smaller or equal
        to our size
    :return: d"""
    d.ts = size
    return d