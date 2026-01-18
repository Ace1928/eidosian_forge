from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
class _InternalBuilderRow(_BuilderRow):
    """The stored state accumulated while writing out internal rows."""

    def finish_node(self, pad=True):
        if not pad:
            raise AssertionError('Must pad internal nodes only.')
        _BuilderRow.finish_node(self)