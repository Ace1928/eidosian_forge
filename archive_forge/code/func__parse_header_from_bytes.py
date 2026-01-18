from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _parse_header_from_bytes(self, bytes):
    """Parse the header from a region of bytes.

        :param bytes: The data to parse.
        :return: An offset, data tuple such as readv yields, for the unparsed
            data. (which may be of length 0).
        """
    signature = bytes[0:len(self._signature())]
    if not signature == self._signature():
        raise index.BadIndexFormatSignature(self._name, BTreeGraphIndex)
    lines = bytes[len(self._signature()):].splitlines()
    options_line = lines[0]
    if not options_line.startswith(_OPTION_NODE_REFS):
        raise index.BadIndexOptions(self)
    try:
        self.node_ref_lists = int(options_line[len(_OPTION_NODE_REFS):])
    except ValueError:
        raise index.BadIndexOptions(self)
    options_line = lines[1]
    if not options_line.startswith(_OPTION_KEY_ELEMENTS):
        raise index.BadIndexOptions(self)
    try:
        self._key_length = int(options_line[len(_OPTION_KEY_ELEMENTS):])
    except ValueError:
        raise index.BadIndexOptions(self)
    options_line = lines[2]
    if not options_line.startswith(_OPTION_LEN):
        raise index.BadIndexOptions(self)
    try:
        self._key_count = int(options_line[len(_OPTION_LEN):])
    except ValueError:
        raise index.BadIndexOptions(self)
    options_line = lines[3]
    if not options_line.startswith(_OPTION_ROW_LENGTHS):
        raise index.BadIndexOptions(self)
    try:
        self._row_lengths = [int(length) for length in options_line[len(_OPTION_ROW_LENGTHS):].split(b',') if length]
    except ValueError:
        raise index.BadIndexOptions(self)
    self._compute_row_offsets()
    header_end = len(signature) + sum(map(len, lines[0:4])) + 4
    return (header_end, bytes[header_end:])