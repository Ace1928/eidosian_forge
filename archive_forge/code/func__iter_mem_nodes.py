from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _iter_mem_nodes(self):
    """Iterate over the nodes held in memory."""
    nodes = self._nodes
    if self.reference_lists:
        for key in sorted(nodes):
            references, value = nodes[key]
            yield (self, key, value, references)
    else:
        for key in sorted(nodes):
            references, value = nodes[key]
            yield (self, key, value)