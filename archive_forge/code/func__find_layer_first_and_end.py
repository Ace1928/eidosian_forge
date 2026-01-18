from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _find_layer_first_and_end(self, offset):
    """Find the start/stop nodes for the layer corresponding to offset.

        :return: (first, end)
            first is the first node in this layer
            end is the first node of the next layer
        """
    first = end = 0
    for roffset in self._row_offsets:
        first = end
        end = roffset
        if offset < roffset:
            break
    return (first, end)