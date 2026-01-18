from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _spill_mem_keys_without_combining(self):
    return self._write_nodes(self._iter_mem_nodes(), allow_optimize=False)