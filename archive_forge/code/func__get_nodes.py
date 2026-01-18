from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _get_nodes(self, cache, node_indexes):
    found = {}
    needed = []
    for idx in node_indexes:
        if idx == 0 and self._root_node is not None:
            found[0] = self._root_node
            continue
        try:
            found[idx] = cache[idx]
        except KeyError:
            needed.append(idx)
    if not needed:
        return found
    needed = self._expand_offsets(needed)
    found.update(self._get_and_cache_nodes(needed))
    return found