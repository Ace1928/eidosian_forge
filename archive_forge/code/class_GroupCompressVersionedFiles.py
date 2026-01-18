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
class GroupCompressVersionedFiles(VersionedFilesWithFallbacks):
    """A group-compress based VersionedFiles implementation."""
    _DEFAULT_MAX_BYTES_TO_INDEX = 1024 * 1024
    _DEFAULT_COMPRESSOR_SETTINGS = {'max_bytes_to_index': _DEFAULT_MAX_BYTES_TO_INDEX}

    def __init__(self, index, access, delta=True, _unadded_refs=None, _group_cache=None):
        """Create a GroupCompressVersionedFiles object.

        :param index: The index object storing access and graph data.
        :param access: The access object storing raw data.
        :param delta: Whether to delta compress or just entropy compress.
        :param _unadded_refs: private parameter, don't use.
        :param _group_cache: private parameter, don't use.
        """
        self._index = index
        self._access = access
        self._delta = delta
        if _unadded_refs is None:
            _unadded_refs = {}
        self._unadded_refs = _unadded_refs
        if _group_cache is None:
            _group_cache = LRUSizeCache(max_size=50 * 1024 * 1024)
        self._group_cache = _group_cache
        self._immediate_fallback_vfs = []
        self._max_bytes_to_index = None

    def without_fallbacks(self):
        """Return a clone of this object without any fallbacks configured."""
        return GroupCompressVersionedFiles(self._index, self._access, self._delta, _unadded_refs=dict(self._unadded_refs), _group_cache=self._group_cache)

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        """Add a text to the store.

        :param key: The key tuple of the text to add.
        :param parents: The parents key tuples of the text to add.
        :param lines: A list of lines. Each line must be a bytestring. And all
            of them except the last must be terminated with \\n and contain no
            other \\n's. The last line may either contain no \\n's or a single
            terminating \\n. If the lines list does meet this constraint the
            add routine may error or may succeed - but you will be unable to
            read the data back accurately. (Checking the lines have been split
            correctly is expensive and extremely unlikely to catch bugs so it
            is not done at runtime unless check_content is True.)
        :param parent_texts: An optional dictionary containing the opaque
            representations of some or all of the parents of version_id to
            allow delta optimisations.  VERY IMPORTANT: the texts must be those
            returned by add_lines or data corruption can be caused.
        :param left_matching_blocks: a hint about which areas are common
            between the text and its left-hand-parent.  The format is
            the SequenceMatcher.get_matching_blocks format.
        :param nostore_sha: Raise ExistingContent and do not add the lines to
            the versioned file if the digest of the lines matches this.
        :param random_id: If True a random id has been selected rather than
            an id determined by some deterministic process such as a converter
            from a foreign VCS. When True the backend may choose not to check
            for uniqueness of the resulting key within the versioned file, so
            this should only be done when the result is expected to be unique
            anyway.
        :param check_content: If True, the lines supplied are verified to be
            bytestrings that are correctly formed lines.
        :return: The text sha1, the number of bytes in the text, and an opaque
                 representation of the inserted version which can be provided
                 back to future add_lines calls in the parent_texts dictionary.
        """
        self._index._check_write_ok()
        if check_content:
            self._check_lines_not_unicode(lines)
            self._check_lines_are_lines(lines)
        return self.add_content(ChunkedContentFactory(key, parents, osutils.sha_strings(lines), lines, chunks_are_lines=True), parent_texts, left_matching_blocks, nostore_sha, random_id)

    def add_content(self, factory, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False):
        """Add a text to the store.

        :param factory: A ContentFactory that can be used to retrieve the key,
            parents and contents.
        :param parent_texts: An optional dictionary containing the opaque
            representations of some or all of the parents of version_id to
            allow delta optimisations.  VERY IMPORTANT: the texts must be those
            returned by add_lines or data corruption can be caused.
        :param left_matching_blocks: a hint about which areas are common
            between the text and its left-hand-parent.  The format is
            the SequenceMatcher.get_matching_blocks format.
        :param nostore_sha: Raise ExistingContent and do not add the lines to
            the versioned file if the digest of the lines matches this.
        :param random_id: If True a random id has been selected rather than
            an id determined by some deterministic process such as a converter
            from a foreign VCS. When True the backend may choose not to check
            for uniqueness of the resulting key within the versioned file, so
            this should only be done when the result is expected to be unique
            anyway.
        :return: The text sha1, the number of bytes in the text, and an opaque
                 representation of the inserted version which can be provided
                 back to future add_lines calls in the parent_texts dictionary.
        """
        self._index._check_write_ok()
        parents = factory.parents
        self._check_add(factory.key, random_id)
        if parents is None:
            parents = ()
        sha1, length = list(self._insert_record_stream([factory], random_id=random_id, nostore_sha=nostore_sha))[0]
        return (sha1, length, None)

    def add_fallback_versioned_files(self, a_versioned_files):
        """Add a source of texts for texts not present in this knit.

        :param a_versioned_files: A VersionedFiles object.
        """
        self._immediate_fallback_vfs.append(a_versioned_files)

    def annotate(self, key):
        """See VersionedFiles.annotate."""
        ann = self.get_annotator()
        return ann.annotate_flat(key)

    def get_annotator(self):
        from ..annotate import Annotator
        return Annotator(self)

    def check(self, progress_bar=None, keys=None):
        """See VersionedFiles.check()."""
        if keys is None:
            keys = self.keys()
            for record in self.get_record_stream(keys, 'unordered', True):
                for chunk in record.iter_bytes_as('chunked'):
                    pass
        else:
            return self.get_record_stream(keys, 'unordered', True)

    def clear_cache(self):
        """See VersionedFiles.clear_cache()"""
        self._group_cache.clear()
        self._index._graph_index.clear_cache()
        self._index._int_cache.clear()

    def _check_add(self, key, random_id):
        """check that version_id and lines are safe to add."""
        version_id = key[-1]
        if version_id is not None:
            if osutils.contains_whitespace(version_id):
                raise errors.InvalidRevisionId(version_id, self)
        self.check_not_reserved_id(version_id)

    def get_parent_map(self, keys):
        """Get a map of the graph parents of keys.

        :param keys: The keys to look up parents for.
        :return: A mapping from keys to parents. Absent keys are absent from
            the mapping.
        """
        return self._get_parent_map_with_sources(keys)[0]

    def _get_parent_map_with_sources(self, keys):
        """Get a map of the parents of keys.

        :param keys: The keys to look up parents for.
        :return: A tuple. The first element is a mapping from keys to parents.
            Absent keys are absent from the mapping. The second element is a
            list with the locations each key was found in. The first element
            is the in-this-knit parents, the second the first fallback source,
            and so on.
        """
        result = {}
        sources = [self._index] + self._immediate_fallback_vfs
        source_results = []
        missing = set(keys)
        for source in sources:
            if not missing:
                break
            new_result = source.get_parent_map(missing)
            source_results.append(new_result)
            result.update(new_result)
            missing.difference_update(set(new_result))
        return (result, source_results)

    def _get_blocks(self, read_memos):
        """Get GroupCompressBlocks for the given read_memos.

        :returns: a series of (read_memo, block) pairs, in the order they were
            originally passed.
        """
        cached = {}
        for read_memo in read_memos:
            try:
                block = self._group_cache[read_memo]
            except KeyError:
                pass
            else:
                cached[read_memo] = block
        not_cached = []
        not_cached_seen = set()
        for read_memo in read_memos:
            if read_memo in cached:
                continue
            if read_memo in not_cached_seen:
                continue
            not_cached.append(read_memo)
            not_cached_seen.add(read_memo)
        raw_records = self._access.get_raw_records(not_cached)
        for read_memo in read_memos:
            try:
                yield (read_memo, cached[read_memo])
            except KeyError:
                zdata = next(raw_records)
                block = GroupCompressBlock.from_bytes(zdata)
                self._group_cache[read_memo] = block
                cached[read_memo] = block
                yield (read_memo, block)

    def get_missing_compression_parent_keys(self):
        """Return the keys of missing compression parents.

        Missing compression parents occur when a record stream was missing
        basis texts, or a index was scanned that had missing basis texts.
        """
        return frozenset()

    def get_record_stream(self, keys, ordering, include_delta_closure):
        """Get a stream of records for keys.

        :param keys: The keys to include.
        :param ordering: Either 'unordered' or 'topological'. A topologically
            sorted stream has compression parents strictly before their
            children.
        :param include_delta_closure: If True then the closure across any
            compression parents will be included (in the opaque data).
        :return: An iterator of ContentFactory objects, each of which is only
            valid until the iterator is advanced.
        """
        orig_keys = list(keys)
        keys = set(keys)
        if not keys:
            return
        if not self._index.has_graph and ordering in ('topological', 'groupcompress'):
            ordering = 'unordered'
        remaining_keys = keys
        while True:
            try:
                keys = set(remaining_keys)
                for content_factory in self._get_remaining_record_stream(keys, orig_keys, ordering, include_delta_closure):
                    remaining_keys.discard(content_factory.key)
                    yield content_factory
                return
            except pack_repo.RetryWithNewPacks as e:
                self._access.reload_or_raise(e)

    def _find_from_fallback(self, missing):
        """Find whatever keys you can from the fallbacks.

        :param missing: A set of missing keys. This set will be mutated as keys
            are found from a fallback_vfs
        :return: (parent_map, key_to_source_map, source_results)
            parent_map  the overall key => parent_keys
            key_to_source_map   a dict from {key: source}
            source_results      a list of (source: keys)
        """
        parent_map = {}
        key_to_source_map = {}
        source_results = []
        for source in self._immediate_fallback_vfs:
            if not missing:
                break
            source_parents = source.get_parent_map(missing)
            parent_map.update(source_parents)
            source_parents = list(source_parents)
            source_results.append((source, source_parents))
            key_to_source_map.update(((key, source) for key in source_parents))
            missing.difference_update(source_parents)
        return (parent_map, key_to_source_map, source_results)

    def _get_ordered_source_keys(self, ordering, parent_map, key_to_source_map):
        """Get the (source, [keys]) list.

        The returned objects should be in the order defined by 'ordering',
        which can weave between different sources.

        :param ordering: Must be one of 'topological' or 'groupcompress'
        :return: List of [(source, [keys])] tuples, such that all keys are in
            the defined order, regardless of source.
        """
        if ordering == 'topological':
            present_keys = tsort.topo_sort(parent_map)
        else:
            present_keys = sort_gc_optimal(parent_map)
        source_keys = []
        current_source = None
        for key in present_keys:
            source = key_to_source_map.get(key, self)
            if source is not current_source:
                source_keys.append((source, []))
                current_source = source
            source_keys[-1][1].append(key)
        return source_keys

    def _get_as_requested_source_keys(self, orig_keys, locations, unadded_keys, key_to_source_map):
        source_keys = []
        current_source = None
        for key in orig_keys:
            if key in locations or key in unadded_keys:
                source = self
            elif key in key_to_source_map:
                source = key_to_source_map[key]
            else:
                continue
            if source is not current_source:
                source_keys.append((source, []))
                current_source = source
            source_keys[-1][1].append(key)
        return source_keys

    def _get_io_ordered_source_keys(self, locations, unadded_keys, source_result):

        def get_group(key):
            return locations[key][0]
        present_keys = list(unadded_keys)
        present_keys.extend(sorted(locations, key=get_group))
        source_keys = [(self, present_keys)]
        source_keys.extend(source_result)
        return source_keys

    def _get_remaining_record_stream(self, keys, orig_keys, ordering, include_delta_closure):
        """Get a stream of records for keys.

        :param keys: The keys to include.
        :param ordering: one of 'unordered', 'topological', 'groupcompress' or
            'as-requested'
        :param include_delta_closure: If True then the closure across any
            compression parents will be included (in the opaque data).
        :return: An iterator of ContentFactory objects, each of which is only
            valid until the iterator is advanced.
        """
        locations = self._index.get_build_details(keys)
        unadded_keys = set(self._unadded_refs).intersection(keys)
        missing = keys.difference(locations)
        missing.difference_update(unadded_keys)
        fallback_parent_map, key_to_source_map, source_result = self._find_from_fallback(missing)
        if ordering in ('topological', 'groupcompress'):
            parent_map = {key: details[2] for key, details in locations.items()}
            for key in unadded_keys:
                parent_map[key] = self._unadded_refs[key]
            parent_map.update(fallback_parent_map)
            source_keys = self._get_ordered_source_keys(ordering, parent_map, key_to_source_map)
        elif ordering == 'as-requested':
            source_keys = self._get_as_requested_source_keys(orig_keys, locations, unadded_keys, key_to_source_map)
        else:
            source_keys = self._get_io_ordered_source_keys(locations, unadded_keys, source_result)
        for key in missing:
            yield AbsentContentFactory(key)
        batcher = _BatchingBlockFetcher(self, locations, get_compressor_settings=self._get_compressor_settings)
        for source, keys in source_keys:
            if source is self:
                for key in keys:
                    if key in self._unadded_refs:
                        yield from batcher.yield_factories(full_flush=True)
                        chunks, sha1 = self._compressor.extract(key)
                        parents = self._unadded_refs[key]
                        yield ChunkedContentFactory(key, parents, sha1, chunks)
                        continue
                    if batcher.add_key(key) > BATCH_SIZE:
                        yield from batcher.yield_factories()
            else:
                yield from batcher.yield_factories(full_flush=True)
                yield from source.get_record_stream(keys, ordering, include_delta_closure)
        yield from batcher.yield_factories(full_flush=True)

    def get_sha1s(self, keys):
        """See VersionedFiles.get_sha1s()."""
        result = {}
        for record in self.get_record_stream(keys, 'unordered', True):
            if record.sha1 is not None:
                result[record.key] = record.sha1
            elif record.storage_kind != 'absent':
                result[record.key] = osutils.sha_strings(record.iter_bytes_as('chunked'))
        return result

    def insert_record_stream(self, stream):
        """Insert a record stream into this container.

        :param stream: A stream of records to insert.
        :return: None
        :seealso VersionedFiles.get_record_stream:
        """
        for _, _ in self._insert_record_stream(stream, random_id=False):
            pass

    def _get_compressor_settings(self):
        from ..config import GlobalConfig
        if self._max_bytes_to_index is None:
            c = GlobalConfig()
            val = c.get_user_option('bzr.groupcompress.max_bytes_to_index')
            if val is not None:
                try:
                    val = int(val)
                except ValueError as e:
                    trace.warning('Value for "bzr.groupcompress.max_bytes_to_index" %r is not an integer' % (val,))
                    val = None
            if val is None:
                val = self._DEFAULT_MAX_BYTES_TO_INDEX
            self._max_bytes_to_index = val
        return {'max_bytes_to_index': self._max_bytes_to_index}

    def _make_group_compressor(self):
        return GroupCompressor(self._get_compressor_settings())

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

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        """Iterate over the lines in the versioned files from keys.

        This may return lines from other keys. Each item the returned
        iterator yields is a tuple of a line and a text version that that line
        is present in (not introduced in).

        Ordering of results is in whatever order is most suitable for the
        underlying storage format.

        If a progress bar is supplied, it may be used to indicate progress.
        The caller is responsible for cleaning up progress bars (because this
        is an iterator).

        NOTES:
         * Lines are normalised by the underlying store: they will all have 

           terminators.
         * Lines are returned in arbitrary order.

        :return: An iterator over (line, key).
        """
        keys = set(keys)
        total = len(keys)
        for key_idx, record in enumerate(self.get_record_stream(keys, 'unordered', True)):
            key = record.key
            if pb is not None:
                pb.update('Walking content', key_idx, total)
            if record.storage_kind == 'absent':
                raise errors.RevisionNotPresent(key, self)
            for line in record.iter_bytes_as('lines'):
                yield (line, key)
        if pb is not None:
            pb.update('Walking content', total, total)

    def keys(self):
        """See VersionedFiles.keys."""
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(2, 'keys scales with size of history')
        sources = [self._index] + self._immediate_fallback_vfs
        result = set()
        for source in sources:
            result.update(source.keys())
        return result