from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _compute_total_pages_in_index(self):
    """How many pages are in the index.

        If we have read the header we will use the value stored there.
        Otherwise it will be computed based on the length of the index.
        """
    if self._size is None:
        raise AssertionError('_compute_total_pages_in_index should not be called when self._size is None')
    if self._root_node is not None:
        return self._row_offsets[-1]
    total_pages = int(math.ceil(self._size / _PAGE_SIZE))
    return total_pages