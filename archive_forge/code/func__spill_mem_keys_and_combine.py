from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _spill_mem_keys_and_combine(self):
    iterators_to_combine = [self._iter_mem_nodes()]
    pos = -1
    for pos, backing in enumerate(self._backing_indices):
        if backing is None:
            pos -= 1
            break
        iterators_to_combine.append(backing.iter_all_entries())
    backing_pos = pos + 1
    new_backing_file, size = self._write_nodes(self._iter_smallest(iterators_to_combine), allow_optimize=False)
    return (new_backing_file, size, backing_pos)