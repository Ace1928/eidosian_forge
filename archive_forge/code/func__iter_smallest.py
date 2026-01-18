from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _iter_smallest(self, iterators_to_combine):
    if len(iterators_to_combine) == 1:
        yield from iterators_to_combine[0]
        return
    current_values = []
    for iterator in iterators_to_combine:
        try:
            current_values.append(next(iterator))
        except StopIteration:
            current_values.append(None)
    last = None
    while True:
        candidates = [(item[1][1], item) for item in enumerate(current_values) if item[1] is not None]
        if not len(candidates):
            return
        selected = min(candidates)
        selected = selected[1]
        if last == selected[1][1]:
            raise index.BadIndexDuplicateKey(last, self)
        last = selected[1][1]
        yield ((self,) + selected[1][1:])
        pos = selected[0]
        try:
            current_values[pos] = next(iterators_to_combine[pos])
        except StopIteration:
            current_values[pos] = None