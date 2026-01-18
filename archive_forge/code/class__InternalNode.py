from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
class _InternalNode:
    """An internal node for a serialised B+Tree index."""
    __slots__ = ('keys', 'offset')

    def __init__(self, bytes):
        """Parse bytes to create an internal node object."""
        self.keys = self._parse_lines(bytes.split(b'\n'))

    def _parse_lines(self, lines):
        nodes = []
        self.offset = int(lines[1][7:])
        as_st = static_tuple.StaticTuple.from_sequence
        for line in lines[2:]:
            if line == b'':
                break
            nodes.append(as_st(line.split(b'\x00')).intern())
        return nodes