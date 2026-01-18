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
class PyrexGroupCompressor(_CommonGroupCompressor):
    """Produce a serialised group of compressed texts.

    It contains code very similar to SequenceMatcher because of having a similar
    task. However some key differences apply:

    * there is no junk, we want a minimal edit not a human readable diff.
    * we don't filter very common lines (because we don't know where a good
      range will start, and after the first text we want to be emitting minmal
      edits only.
    * we chain the left side, not the right side
    * we incrementally update the adjacency matrix as new lines are provided.
    * we look for matches in all of the left side, so the routine which does
      the analagous task of find_longest_match does not need to filter on the
      left side.
    """

    def __init__(self, settings=None):
        super().__init__(settings)
        max_bytes_to_index = self._settings.get('max_bytes_to_index', 0)
        self._delta_index = DeltaIndex(max_bytes_to_index=max_bytes_to_index)

    def _compress(self, key, chunks, input_len, max_delta_size, soft=False):
        """see _CommonGroupCompressor._compress"""
        if self._delta_index._source_offset != self.endpoint:
            raise AssertionError('_source_offset != endpoint somehow the DeltaIndex got out of sync with the output lines')
        bytes = b''.join(chunks)
        delta = self._delta_index.make_delta(bytes, max_delta_size)
        if delta is None:
            type = 'fulltext'
            enc_length = encode_base128_int(input_len)
            len_mini_header = 1 + len(enc_length)
            self._delta_index.add_source(bytes, len_mini_header)
            new_chunks = [b'f', enc_length] + chunks
        else:
            type = 'delta'
            enc_length = encode_base128_int(len(delta))
            len_mini_header = 1 + len(enc_length)
            new_chunks = [b'd', enc_length, delta]
            self._delta_index.add_delta_source(delta, len_mini_header)
        start = self.endpoint
        chunk_start = len(self.chunks)
        self._output_chunks(new_chunks)
        self.input_bytes += input_len
        chunk_end = len(self.chunks)
        self.labels_deltas[key] = (start, chunk_start, self.endpoint, chunk_end)
        if not self._delta_index._source_offset == self.endpoint:
            raise AssertionError('the delta index is out of syncwith the output lines %s != %s' % (self._delta_index._source_offset, self.endpoint))
        return (start, self.endpoint, type)

    def _output_chunks(self, new_chunks):
        """Output some chunks.

        :param new_chunks: The chunks to output.
        """
        self._last = (len(self.chunks), self.endpoint)
        endpoint = self.endpoint
        self.chunks.extend(new_chunks)
        endpoint += sum(map(len, new_chunks))
        self.endpoint = endpoint