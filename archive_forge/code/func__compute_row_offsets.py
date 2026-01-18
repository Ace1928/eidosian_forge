from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _compute_row_offsets(self):
    """Fill out the _row_offsets attribute based on _row_lengths."""
    offsets = []
    row_offset = 0
    for row in self._row_lengths:
        offsets.append(row_offset)
        row_offset += row
    offsets.append(row_offset)
    self._row_offsets = offsets