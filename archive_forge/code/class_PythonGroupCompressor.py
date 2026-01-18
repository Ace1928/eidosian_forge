import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
class PythonGroupCompressor(_CommonGroupCompressor):

    def __init__(self, settings=None):
        """Create a GroupCompressor.

        Used only if the pyrex version is not available.
        """
        super().__init__(settings)
        self._delta_index = LinesDeltaIndex([])
        self.chunks = self._delta_index.lines

    def _compress(self, key, chunks, input_len, max_delta_size, soft=False):
        """see _CommonGroupCompressor._compress"""
        new_lines = osutils.chunks_to_lines(chunks)
        out_lines, index_lines = self._delta_index.make_delta(new_lines, bytes_length=input_len, soft=soft)
        delta_length = sum(map(len, out_lines))
        if delta_length > max_delta_size:
            type = 'fulltext'
            out_lines = [b'f', encode_base128_int(input_len)]
            out_lines.extend(new_lines)
            index_lines = [False, False]
            index_lines.extend([True] * len(new_lines))
        else:
            type = 'delta'
            out_lines[0] = b'd'
            out_lines[1] = encode_base128_int(delta_length)
        start = self.endpoint
        chunk_start = len(self.chunks)
        self._last = (chunk_start, self.endpoint)
        self._delta_index.extend_lines(out_lines, index_lines)
        self.endpoint = self._delta_index.endpoint
        self.input_bytes += input_len
        chunk_end = len(self.chunks)
        self.labels_deltas[key] = (start, chunk_start, self.endpoint, chunk_end)
        return (start, self.endpoint, type)