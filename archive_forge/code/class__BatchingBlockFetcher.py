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
class _BatchingBlockFetcher:
    """Fetch group compress blocks in batches.

    :ivar total_bytes: int of expected number of bytes needed to fetch the
        currently pending batch.
    """

    def __init__(self, gcvf, locations, get_compressor_settings=None):
        self.gcvf = gcvf
        self.locations = locations
        self.keys = []
        self.batch_memos = {}
        self.memos_to_get = []
        self.total_bytes = 0
        self.last_read_memo = None
        self.manager = None
        self._get_compressor_settings = get_compressor_settings

    def add_key(self, key):
        """Add another to key to fetch.

        :return: The estimated number of bytes needed to fetch the batch so
            far.
        """
        self.keys.append(key)
        index_memo, _, _, _ = self.locations[key]
        read_memo = index_memo[0:3]
        if read_memo in self.batch_memos:
            return self.total_bytes
        try:
            cached_block = self.gcvf._group_cache[read_memo]
        except KeyError:
            self.batch_memos[read_memo] = None
            self.memos_to_get.append(read_memo)
            byte_length = read_memo[2]
            self.total_bytes += byte_length
        else:
            self.batch_memos[read_memo] = cached_block
        return self.total_bytes

    def _flush_manager(self):
        if self.manager is not None:
            yield from self.manager.get_record_stream()
            self.manager = None
            self.last_read_memo = None

    def yield_factories(self, full_flush=False):
        """Yield factories for keys added since the last yield.  They will be
        returned in the order they were added via add_key.

        :param full_flush: by default, some results may not be returned in case
            they can be part of the next batch.  If full_flush is True, then
            all results are returned.
        """
        if self.manager is None and (not self.keys):
            return
        blocks = self.gcvf._get_blocks(self.memos_to_get)
        memos_to_get_stack = list(self.memos_to_get)
        memos_to_get_stack.reverse()
        for key in self.keys:
            index_memo, _, parents, _ = self.locations[key]
            read_memo = index_memo[:3]
            if self.last_read_memo != read_memo:
                yield from self._flush_manager()
                if memos_to_get_stack and memos_to_get_stack[-1] == read_memo:
                    block_read_memo, block = next(blocks)
                    if block_read_memo != read_memo:
                        raise AssertionError('block_read_memo out of sync with read_memo(%r != %r)' % (block_read_memo, read_memo))
                    self.batch_memos[read_memo] = block
                    memos_to_get_stack.pop()
                else:
                    block = self.batch_memos[read_memo]
                self.manager = _LazyGroupContentManager(block, get_compressor_settings=self._get_compressor_settings)
                self.last_read_memo = read_memo
            start, end = index_memo[3:5]
            self.manager.add_factory(key, parents, start, end)
        if full_flush:
            yield from self._flush_manager()
        del self.keys[:]
        self.batch_memos.clear()
        del self.memos_to_get[:]
        self.total_bytes = 0