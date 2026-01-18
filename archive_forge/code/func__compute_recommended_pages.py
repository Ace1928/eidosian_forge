from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _compute_recommended_pages(self):
    """Convert transport's recommended_page_size into btree pages.

        recommended_page_size is in bytes, we want to know how many _PAGE_SIZE
        pages fit in that length.
        """
    recommended_read = self._transport.recommended_page_size()
    recommended_pages = int(math.ceil(recommended_read / _PAGE_SIZE))
    return recommended_pages