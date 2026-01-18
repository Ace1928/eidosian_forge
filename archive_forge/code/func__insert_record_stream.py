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
def _insert_record_stream(self, stream, random_id=False, nostore_sha=None, reuse_blocks=True):
    """Internal core to insert a record stream into this container.

        This helper function has a different interface than insert_record_stream
        to allow add_lines to be minimal, but still return the needed data.

        :param stream: A stream of records to insert.
        :param nostore_sha: If the sha1 of a given text matches nostore_sha,
            raise ExistingContent, rather than committing the new text.
        :param reuse_blocks: If the source is streaming from
            groupcompress-blocks, just insert the blocks as-is, rather than
            expanding the texts and inserting again.
        :return: An iterator over (sha1, length) of the inserted records.
        :seealso insert_record_stream:
        :seealso add_lines:
        """
    adapters = {}

    def get_adapter(adapter_key):
        try:
            return adapters[adapter_key]
        except KeyError:
            adapter_factory = adapter_registry.get(adapter_key)
            adapter = adapter_factory(self)
            adapters[adapter_key] = adapter
            return adapter
    self._compressor = self._make_group_compressor()
    self._unadded_refs = {}
    keys_to_add = []

    def flush():
        bytes_len, chunks = self._compressor.flush().to_chunks()
        self._compressor = self._make_group_compressor()
        index, start, length = self._access.add_raw_record(None, bytes_len, chunks)
        nodes = []
        for key, reads, refs in keys_to_add:
            nodes.append((key, b'%d %d %s' % (start, length, reads), refs))
        self._index.add_records(nodes, random_id=random_id)
        self._unadded_refs = {}
        del keys_to_add[:]
    last_prefix = None
    max_fulltext_len = 0
    max_fulltext_prefix = None
    insert_manager = None
    block_start = None
    block_length = None
    inserted_keys = set()
    reuse_this_block = reuse_blocks
    for record in stream:
        if record.storage_kind == 'absent':
            raise errors.RevisionNotPresent(record.key, self)
        if random_id:
            if record.key in inserted_keys:
                trace.note(gettext('Insert claimed random_id=True, but then inserted %r two times'), record.key)
                continue
            inserted_keys.add(record.key)
        if reuse_blocks:
            if record.storage_kind == 'groupcompress-block':
                insert_manager = record._manager
                reuse_this_block = insert_manager.check_is_well_utilized()
        else:
            reuse_this_block = False
        if reuse_this_block:
            if record.storage_kind == 'groupcompress-block':
                insert_manager = record._manager
                bytes_len, chunks = record._manager._block.to_chunks()
                _, start, length = self._access.add_raw_record(None, bytes_len, chunks)
                block_start = start
                block_length = length
            if record.storage_kind in ('groupcompress-block', 'groupcompress-block-ref'):
                if insert_manager is None:
                    raise AssertionError('No insert_manager set')
                if insert_manager is not record._manager:
                    raise AssertionError('insert_manager does not match the current record, we cannot be positive that the appropriate content was inserted.')
                value = b'%d %d %d %d' % (block_start, block_length, record._start, record._end)
                nodes = [(record.key, value, (record.parents,))]
                self._index.add_records(nodes, random_id=random_id)
                continue
        try:
            chunks = record.get_bytes_as('chunked')
        except UnavailableRepresentation:
            adapter_key = (record.storage_kind, 'chunked')
            adapter = get_adapter(adapter_key)
            chunks = adapter.get_bytes(record, 'chunked')
        chunks_len = record.size
        if chunks_len is None:
            chunks_len = sum(map(len, chunks))
        if len(record.key) > 1:
            prefix = record.key[0]
            soft = prefix == last_prefix
        else:
            prefix = None
            soft = False
        if max_fulltext_len < chunks_len:
            max_fulltext_len = chunks_len
            max_fulltext_prefix = prefix
        found_sha1, start_point, end_point, type = self._compressor.compress(record.key, chunks, chunks_len, record.sha1, soft=soft, nostore_sha=nostore_sha)
        if prefix == max_fulltext_prefix and end_point < 2 * max_fulltext_len:
            start_new_block = False
        elif end_point > 4 * 1024 * 1024:
            start_new_block = True
        elif prefix is not None and prefix != last_prefix and (end_point > 2 * 1024 * 1024):
            start_new_block = True
        else:
            start_new_block = False
        last_prefix = prefix
        if start_new_block:
            self._compressor.pop_last()
            flush()
            max_fulltext_len = chunks_len
            found_sha1, start_point, end_point, type = self._compressor.compress(record.key, chunks, chunks_len, record.sha1)
        if record.key[-1] is None:
            key = record.key[:-1] + (b'sha1:' + found_sha1,)
        else:
            key = record.key
        self._unadded_refs[key] = record.parents
        yield (found_sha1, chunks_len)
        as_st = static_tuple.StaticTuple.from_sequence
        if record.parents is not None:
            parents = as_st([as_st(p) for p in record.parents])
        else:
            parents = None
        refs = static_tuple.StaticTuple(parents)
        keys_to_add.append((key, b'%d %d' % (start_point, end_point), refs))
    if len(keys_to_add):
        flush()
    self._compressor = None