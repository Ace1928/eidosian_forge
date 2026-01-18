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
def _iter_bytes_to_values(self, byte_rows):
    """
        Iterator that yields each scanline;
        each scanline being a sequence of values.
        `byte_rows` should be an iterator that yields
        the bytes of each row in turn.
        """
    for row in byte_rows:
        yield self._bytes_to_values(row)