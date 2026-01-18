import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
class KnitVersionedFiles(VersionedFilesWithFallbacks):
    """Storage for many versioned files using knit compression.

    Backend storage is managed by indices and data objects.

    :ivar _index: A _KnitGraphIndex or similar that can describe the
        parents, graph, compression and data location of entries in this
        KnitVersionedFiles.  Note that this is only the index for
        *this* vfs; if there are fallbacks they must be queried separately.
    """

    def __init__(self, index, data_access, max_delta_chain=200, annotated=False, reload_func=None):
        """Create a KnitVersionedFiles with index and data_access.

        :param index: The index for the knit data.
        :param data_access: The access object to store and retrieve knit
            records.
        :param max_delta_chain: The maximum number of deltas to permit during
            insertion. Set to 0 to prohibit the use of deltas.
        :param annotated: Set to True to cause annotations to be calculated and
            stored during insertion.
        :param reload_func: An function that can be called if we think we need
            to reload the pack listing and try again. See
            'breezy.bzr.pack_repo.AggregateIndex' for the signature.
        """
        self._index = index
        self._access = data_access
        self._max_delta_chain = max_delta_chain
        if annotated:
            self._factory = KnitAnnotateFactory()
        else:
            self._factory = KnitPlainFactory()
        self._immediate_fallback_vfs = []
        self._reload_func = reload_func

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._index, self._access)

    def without_fallbacks(self):
        """Return a clone of this object without any fallbacks configured."""
        return KnitVersionedFiles(self._index, self._access, self._max_delta_chain, self._factory.annotated, self._reload_func)

    def add_fallback_versioned_files(self, a_versioned_files):
        """Add a source of texts for texts not present in this knit.

        :param a_versioned_files: A VersionedFiles object.
        """
        self._immediate_fallback_vfs.append(a_versioned_files)

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        """See VersionedFiles.add_lines()."""
        self._index._check_write_ok()
        self._check_add(key, lines, random_id, check_content)
        if parents is None:
            parents = ()
        line_bytes = b''.join(lines)
        return self._add(key, lines, parents, parent_texts, left_matching_blocks, nostore_sha, random_id, line_bytes=line_bytes)

    def add_content(self, content_factory, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False):
        """See VersionedFiles.add_content()."""
        self._index._check_write_ok()
        key = content_factory.key
        parents = content_factory.parents
        self._check_add(key, None, random_id, check_content=False)
        if parents is None:
            parents = ()
        lines = content_factory.get_bytes_as('lines')
        line_bytes = content_factory.get_bytes_as('fulltext')
        return self._add(key, lines, parents, parent_texts, left_matching_blocks, nostore_sha, random_id, line_bytes=line_bytes)

    def _add(self, key, lines, parents, parent_texts, left_matching_blocks, nostore_sha, random_id, line_bytes):
        """Add a set of lines on top of version specified by parents.

        Any versions not present will be converted into ghosts.

        :param lines: A list of strings where each one is a single line (has a
            single newline at the end of the string) This is now optional
            (callers can pass None). It is left in its location for backwards
            compatibility. It should ''.join(lines) must == line_bytes
        :param line_bytes: A single string containing the content

        We pass both lines and line_bytes because different routes bring the
        values to this function. And for memory efficiency, we don't want to
        have to split/join on-demand.
        """
        digest = sha_string(line_bytes)
        if nostore_sha == digest:
            raise ExistingContent
        present_parents = []
        if parent_texts is None:
            parent_texts = {}
        present_parent_map = self._index.get_parent_map(parents)
        for parent in parents:
            if parent in present_parent_map:
                present_parents.append(parent)
        if len(present_parents) == 0 or present_parents[0] != parents[0]:
            delta = False
        else:
            delta = self._check_should_delta(present_parents[0])
        text_length = len(line_bytes)
        options = []
        no_eol = False
        if line_bytes and (not line_bytes.endswith(b'\n')):
            options.append(b'no-eol')
            no_eol = True
            if lines is None:
                lines = osutils.split_lines(line_bytes)
            else:
                lines = lines[:]
            lines[-1] = lines[-1] + b'\n'
        if lines is None:
            lines = osutils.split_lines(line_bytes)
        for element in key[:-1]:
            if not isinstance(element, bytes):
                raise TypeError('key contains non-bytestrings: {!r}'.format(key))
        if key[-1] is None:
            key = key[:-1] + (b'sha1:' + digest,)
        elif not isinstance(key[-1], bytes):
            raise TypeError('key contains non-bytestrings: {!r}'.format(key))
        version_id = key[-1]
        content = self._factory.make(lines, version_id)
        if no_eol:
            content._should_strip_eol = True
        if delta or (self._factory.annotated and len(present_parents) > 0):
            delta_hunks = self._merge_annotations(content, present_parents, parent_texts, delta, self._factory.annotated, left_matching_blocks)
        if delta:
            options.append(b'line-delta')
            store_lines = self._factory.lower_line_delta(delta_hunks)
            size, data = self._record_to_data(key, digest, store_lines)
        else:
            options.append(b'fulltext')
            if self._factory.__class__ is KnitPlainFactory:
                dense_lines = [line_bytes]
                if no_eol:
                    dense_lines.append(b'\n')
                size, data = self._record_to_data(key, digest, lines, dense_lines)
            else:
                store_lines = self._factory.lower_fulltext(content)
                size, data = self._record_to_data(key, digest, store_lines)
        access_memo = self._access.add_raw_record(key, size, data)
        self._index.add_records(((key, options, access_memo, parents),), random_id=random_id)
        return (digest, text_length, content)

    def annotate(self, key):
        """See VersionedFiles.annotate."""
        return self._factory.annotate(self, key)

    def get_annotator(self):
        return _KnitAnnotator(self)

    def check(self, progress_bar=None, keys=None):
        """See VersionedFiles.check()."""
        if keys is None:
            return self._logical_check()
        else:
            return self.get_record_stream(keys, 'unordered', True)

    def _logical_check(self):
        keys = self._index.keys()
        parent_map = self.get_parent_map(keys)
        for key in keys:
            if self._index.get_method(key) != 'fulltext':
                compression_parent = parent_map[key][0]
                if compression_parent not in parent_map:
                    raise KnitCorrupt(self, 'Missing basis parent {} for {}'.format(compression_parent, key))
        for fallback_vfs in self._immediate_fallback_vfs:
            fallback_vfs.check()

    def _check_add(self, key, lines, random_id, check_content):
        """check that version_id and lines are safe to add."""
        if not all((isinstance(x, bytes) or x is None for x in key)):
            raise TypeError(key)
        version_id = key[-1]
        if version_id is not None:
            if contains_whitespace(version_id):
                raise InvalidRevisionId(version_id, self)
            self.check_not_reserved_id(version_id)
        if check_content:
            self._check_lines_not_unicode(lines)
            self._check_lines_are_lines(lines)

    def _check_header(self, key, line):
        rec = self._split_header(line)
        self._check_header_version(rec, key[-1])
        return rec

    def _check_header_version(self, rec, version_id):
        """Checks the header version on original format knit records.

        These have the last component of the key embedded in the record.
        """
        if rec[1] != version_id:
            raise KnitCorrupt(self, 'unexpected version, wanted {!r}, got {!r}'.format(version_id, rec[1]))

    def _check_should_delta(self, parent):
        """Iterate back through the parent listing, looking for a fulltext.

        This is used when we want to decide whether to add a delta or a new
        fulltext. It searches for _max_delta_chain parents. When it finds a
        fulltext parent, it sees if the total size of the deltas leading up to
        it is large enough to indicate that we want a new full text anyway.

        Return True if we should create a new delta, False if we should use a
        full text.
        """
        delta_size = 0
        fulltext_size = None
        for count in range(self._max_delta_chain):
            try:
                build_details = self._index.get_build_details([parent])
                parent_details = build_details[parent]
            except (RevisionNotPresent, KeyError) as e:
                return False
            index_memo, compression_parent, _, _ = parent_details
            _, _, size = index_memo
            if compression_parent is None:
                fulltext_size = size
                break
            delta_size += size
            parent = compression_parent
        else:
            return False
        return fulltext_size > delta_size

    def _build_details_to_components(self, build_details):
        """Convert a build_details tuple to a position tuple."""
        return (build_details[3], build_details[0], build_details[1])

    def _get_components_positions(self, keys, allow_missing=False):
        """Produce a map of position data for the components of keys.

        This data is intended to be used for retrieving the knit records.

        A dict of key to (record_details, index_memo, next, parents) is
        returned.

        * method is the way referenced data should be applied.
        * index_memo is the handle to pass to the data access to actually get
          the data
        * next is the build-parent of the version, or None for fulltexts.
        * parents is the version_ids of the parents of this version

        :param allow_missing: If True do not raise an error on a missing
            component, just ignore it.
        """
        component_data = {}
        pending_components = keys
        while pending_components:
            build_details = self._index.get_build_details(pending_components)
            current_components = set(pending_components)
            pending_components = set()
            for key, details in build_details.items():
                index_memo, compression_parent, parents, record_details = details
                if compression_parent is not None:
                    pending_components.add(compression_parent)
                component_data[key] = self._build_details_to_components(details)
            missing = current_components.difference(build_details)
            if missing and (not allow_missing):
                raise errors.RevisionNotPresent(missing.pop(), self)
        return component_data

    def _get_content(self, key, parent_texts={}):
        """Returns a content object that makes up the specified
        version."""
        cached_version = parent_texts.get(key, None)
        if cached_version is not None:
            if not self.get_parent_map([key]):
                raise RevisionNotPresent(key, self)
            return cached_version
        generator = _VFContentMapGenerator(self, [key])
        return generator._get_content(key)

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

    def _get_record_map(self, keys, allow_missing=False):
        """Produce a dictionary of knit records.

        :return: {key:(record, record_details, digest, next)}

            * record: data returned from read_records (a KnitContentobject)
            * record_details: opaque information to pass to parse_record
            * digest: SHA1 digest of the full text after all steps are done
            * next: build-parent of the version, i.e. the leftmost ancestor.
                Will be None if the record is not a delta.

        :param keys: The keys to build a map for
        :param allow_missing: If some records are missing, rather than
            error, just return the data that could be generated.
        """
        raw_map = self._get_record_map_unparsed(keys, allow_missing=allow_missing)
        return self._raw_map_to_record_map(raw_map)

    def _raw_map_to_record_map(self, raw_map):
        """Parse the contents of _get_record_map_unparsed.

        :return: see _get_record_map.
        """
        result = {}
        for key in raw_map:
            data, record_details, next = raw_map[key]
            content, digest = self._parse_record(key[-1], data)
            result[key] = (content, record_details, digest, next)
        return result

    def _get_record_map_unparsed(self, keys, allow_missing=False):
        """Get the raw data for reconstructing keys without parsing it.

        :return: A dict suitable for parsing via _raw_map_to_record_map.
            key-> raw_bytes, (method, noeol), compression_parent
        """
        while True:
            try:
                position_map = self._get_components_positions(keys, allow_missing=allow_missing)
                records = [(key, i_m) for key, (r, i_m, n) in position_map.items()]
                records.sort(key=operator.itemgetter(1))
                raw_record_map = {}
                for key, data in self._read_records_iter_unchecked(records):
                    record_details, index_memo, next = position_map[key]
                    raw_record_map[key] = (data, record_details, next)
                return raw_record_map
            except pack_repo.RetryWithNewPacks as e:
                self._access.reload_or_raise(e)

    @classmethod
    def _split_by_prefix(cls, keys):
        """For the given keys, split them up based on their prefix.

        To keep memory pressure somewhat under control, split the
        requests back into per-file-id requests, otherwise "bzr co"
        extracts the full tree into memory before writing it to disk.
        This should be revisited if _get_content_maps() can ever cross
        file-id boundaries.

        The keys for a given file_id are kept in the same relative order.
        Ordering between file_ids is not, though prefix_order will return the
        order that the key was first seen.

        :param keys: An iterable of key tuples
        :return: (split_map, prefix_order)
            split_map       A dictionary mapping prefix => keys
            prefix_order    The order that we saw the various prefixes
        """
        split_by_prefix = {}
        prefix_order = []
        for key in keys:
            if len(key) == 1:
                prefix = b''
            else:
                prefix = key[0]
            if prefix in split_by_prefix:
                split_by_prefix[prefix].append(key)
            else:
                split_by_prefix[prefix] = [key]
                prefix_order.append(prefix)
        return (split_by_prefix, prefix_order)

    def _group_keys_for_io(self, keys, non_local_keys, positions, _min_buffer_size=_STREAM_MIN_BUFFER_SIZE):
        """For the given keys, group them into 'best-sized' requests.

        The idea is to avoid making 1 request per file, but to never try to
        unpack an entire 1.5GB source tree in a single pass. Also when
        possible, we should try to group requests to the same pack file
        together.

        :return: list of (keys, non_local) tuples that indicate what keys
            should be fetched next.
        """
        total_keys = len(keys)
        prefix_split_keys, prefix_order = self._split_by_prefix(keys)
        prefix_split_non_local_keys, _ = self._split_by_prefix(non_local_keys)
        cur_keys = []
        cur_non_local = set()
        cur_size = 0
        result = []
        sizes = []
        for prefix in prefix_order:
            keys = prefix_split_keys[prefix]
            non_local = prefix_split_non_local_keys.get(prefix, [])
            this_size = self._index._get_total_build_size(keys, positions)
            cur_size += this_size
            cur_keys.extend(keys)
            cur_non_local.update(non_local)
            if cur_size > _min_buffer_size:
                result.append((cur_keys, cur_non_local))
                sizes.append(cur_size)
                cur_keys = []
                cur_non_local = set()
                cur_size = 0
        if cur_keys:
            result.append((cur_keys, cur_non_local))
            sizes.append(cur_size)
        return result

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
        keys = set(keys)
        if not keys:
            return
        if not self._index.has_graph:
            ordering = 'unordered'
        remaining_keys = keys
        while True:
            try:
                keys = set(remaining_keys)
                for content_factory in self._get_remaining_record_stream(keys, ordering, include_delta_closure):
                    remaining_keys.discard(content_factory.key)
                    yield content_factory
                return
            except pack_repo.RetryWithNewPacks as e:
                self._access.reload_or_raise(e)

    def _get_remaining_record_stream(self, keys, ordering, include_delta_closure):
        """This function is the 'retry' portion for get_record_stream."""
        if include_delta_closure:
            positions = self._get_components_positions(keys, allow_missing=True)
        else:
            build_details = self._index.get_build_details(keys)
            positions = {key: self._build_details_to_components(details) for key, details in build_details.items()}
        absent_keys = keys.difference(set(positions))
        if include_delta_closure:
            needed_from_fallback = set()
            reconstructable_keys = {}
            for key in keys:
                try:
                    chain = [key, positions[key][2]]
                except KeyError:
                    needed_from_fallback.add(key)
                    continue
                result = True
                while chain[-1] is not None:
                    if chain[-1] in reconstructable_keys:
                        result = reconstructable_keys[chain[-1]]
                        break
                    else:
                        try:
                            chain.append(positions[chain[-1]][2])
                        except KeyError:
                            needed_from_fallback.add(chain[-1])
                            result = True
                            break
                for chain_key in chain[:-1]:
                    reconstructable_keys[chain_key] = result
                if not result:
                    needed_from_fallback.add(key)
        global_map, parent_maps = self._get_parent_map_with_sources(keys)
        if ordering in ('topological', 'groupcompress'):
            if ordering == 'topological':
                present_keys = tsort.topo_sort(global_map)
            else:
                present_keys = sort_groupcompress(global_map)
            source_keys = []
            current_source = None
            for key in present_keys:
                for parent_map in parent_maps:
                    if key in parent_map:
                        key_source = parent_map
                        break
                if current_source is not key_source:
                    source_keys.append((key_source, []))
                    current_source = key_source
                source_keys[-1][1].append(key)
        else:
            if ordering != 'unordered':
                raise AssertionError('valid values for ordering are: "unordered", "groupcompress" or "topological" not: %r' % (ordering,))
            present_keys = []
            source_keys = []
            for parent_map in reversed(parent_maps):
                source_keys.append((parent_map, []))
                for key in parent_map:
                    present_keys.append(key)
                    source_keys[-1][1].append(key)
            for source, sub_keys in source_keys:
                if source is parent_maps[0]:
                    self._index._sort_keys_by_io(sub_keys, positions)
        absent_keys = keys - set(global_map)
        for key in absent_keys:
            yield AbsentContentFactory(key)
        if include_delta_closure:
            non_local_keys = needed_from_fallback - absent_keys
            for keys, non_local_keys in self._group_keys_for_io(present_keys, non_local_keys, positions):
                generator = _VFContentMapGenerator(self, keys, non_local_keys, global_map, ordering=ordering)
                yield from generator.get_record_stream()
        else:
            for source, keys in source_keys:
                if source is parent_maps[0]:
                    records = [(key, positions[key][1]) for key in keys]
                    for key, raw_data in self._read_records_iter_unchecked(records):
                        record_details, index_memo, _ = positions[key]
                        yield KnitContentFactory(key, global_map[key], record_details, None, raw_data, self._factory.annotated, None)
                else:
                    vf = self._immediate_fallback_vfs[parent_maps.index(source) - 1]
                    yield from vf.get_record_stream(keys, ordering, include_delta_closure)

    def get_sha1s(self, keys):
        """See VersionedFiles.get_sha1s()."""
        missing = set(keys)
        record_map = self._get_record_map(missing, allow_missing=True)
        result = {}
        for key, details in record_map.items():
            if key not in missing:
                continue
            result[key] = details[2]
        missing.difference_update(set(result))
        for source in self._immediate_fallback_vfs:
            if not missing:
                break
            new_result = source.get_sha1s(missing)
            result.update(new_result)
            missing.difference_update(set(new_result))
        return result

    def insert_record_stream(self, stream):
        """Insert a record stream into this container.

        :param stream: A stream of records to insert.
        :return: None
        :seealso VersionedFiles.get_record_stream:
        """

        def get_adapter(adapter_key):
            try:
                return adapters[adapter_key]
            except KeyError:
                adapter_factory = adapter_registry.get(adapter_key)
                adapter = adapter_factory(self)
                adapters[adapter_key] = adapter
                return adapter
        delta_types = set()
        if self._factory.annotated:
            annotated = 'annotated-'
            convertibles = []
        else:
            annotated = ''
            convertibles = {'knit-annotated-ft-gz'}
            if self._max_delta_chain:
                delta_types.add('knit-annotated-delta-gz')
                convertibles.add('knit-annotated-delta-gz')
        native_types = set()
        if self._max_delta_chain:
            native_types.add('knit-%sdelta-gz' % annotated)
            delta_types.add('knit-%sdelta-gz' % annotated)
        native_types.add('knit-%sft-gz' % annotated)
        knit_types = native_types.union(convertibles)
        adapters = {}
        buffered_index_entries = {}
        for record in stream:
            kind = record.storage_kind
            if kind.startswith('knit-') and kind.endswith('-gz'):
                raw_data = record._raw_record
                df, rec = self._parse_record_header(record.key, raw_data)
                df.close()
            buffered = False
            parents = record.parents
            if record.storage_kind in delta_types:
                compression_parent = parents[0]
            else:
                compression_parent = None
            if record.storage_kind == 'absent':
                raise RevisionNotPresent([record.key], self)
            elif record.storage_kind in knit_types and (compression_parent is None or not self._immediate_fallback_vfs or compression_parent in self._index or (compression_parent not in self)):
                if record.storage_kind not in native_types:
                    try:
                        adapter_key = (record.storage_kind, 'knit-delta-gz')
                        adapter = get_adapter(adapter_key)
                    except KeyError:
                        adapter_key = (record.storage_kind, 'knit-ft-gz')
                        adapter = get_adapter(adapter_key)
                    bytes = adapter.get_bytes(record, adapter_key[1])
                else:
                    bytes = record._raw_record
                options = [record._build_details[0].encode('ascii')]
                if record._build_details[1]:
                    options.append(b'no-eol')
                access_memo = self._access.add_raw_record(record.key, len(bytes), [bytes])
                index_entry = (record.key, options, access_memo, parents)
                if b'fulltext' not in options:
                    if compression_parent not in self._index:
                        pending = buffered_index_entries.setdefault(compression_parent, [])
                        pending.append(index_entry)
                        buffered = True
                if not buffered:
                    self._index.add_records([index_entry])
            elif record.storage_kind in ('chunked', 'file'):
                self.add_lines(record.key, parents, record.get_bytes_as('lines'))
            else:
                self._access.flush()
                try:
                    lines = record.get_bytes_as('lines')
                except UnavailableRepresentation:
                    adapter_key = (record.storage_kind, 'lines')
                    adapter = get_adapter(adapter_key)
                    lines = adapter.get_bytes(record, 'lines')
                try:
                    self.add_lines(record.key, parents, lines)
                except errors.RevisionAlreadyPresent:
                    pass
            if not buffered:
                added_keys = [record.key]
                while added_keys:
                    key = added_keys.pop(0)
                    if key in buffered_index_entries:
                        index_entries = buffered_index_entries[key]
                        self._index.add_records(index_entries)
                        added_keys.extend([index_entry[0] for index_entry in index_entries])
                        del buffered_index_entries[key]
        if buffered_index_entries:
            all_entries = []
            for key in buffered_index_entries:
                index_entries = buffered_index_entries[key]
                all_entries.extend(index_entries)
            self._index.add_records(all_entries, missing_compression_parents=True)

    def get_missing_compression_parent_keys(self):
        """Return an iterable of keys of missing compression parents.

        Check this after calling insert_record_stream to find out if there are
        any missing compression parents.  If there are, the records that
        depend on them are not able to be inserted safely. For atomic
        KnitVersionedFiles built on packs, the transaction should be aborted or
        suspended - commit will fail at this point. Nonatomic knits will error
        earlier because they have no staging area to put pending entries into.
        """
        return self._index.get_missing_compression_parents()

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
         * Lines are normalised by the underlying store: they will all have \\n
           terminators.
         * Lines are returned in arbitrary order.
         * If a requested key did not change any lines (or didn't have any
           lines), it may not be mentioned at all in the result.

        :param pb: Progress bar supplied by caller.
        :return: An iterator over (line, key).
        """
        if pb is None:
            pb = ui.ui_factory.nested_progress_bar()
        keys = set(keys)
        total = len(keys)
        done = False
        while not done:
            try:
                key_records = []
                build_details = self._index.get_build_details(keys)
                for key, details in build_details.items():
                    if key in keys:
                        key_records.append((key, details[0]))
                records_iter = enumerate(self._read_records_iter(key_records))
                for key_idx, (key, data, sha_value) in records_iter:
                    pb.update(gettext('Walking content'), key_idx, total)
                    compression_parent = build_details[key][1]
                    if compression_parent is None:
                        line_iterator = self._factory.get_fulltext_content(data)
                    else:
                        line_iterator = self._factory.get_linedelta_content(data)
                    keys.remove(key)
                    for line in line_iterator:
                        yield (line, key)
                done = True
            except pack_repo.RetryWithNewPacks as e:
                self._access.reload_or_raise(e)
        if keys and (not self._immediate_fallback_vfs):
            raise RevisionNotPresent(keys, repr(self))
        for source in self._immediate_fallback_vfs:
            if not keys:
                break
            source_keys = set()
            for line, key in source.iter_lines_added_or_present_in_keys(keys):
                source_keys.add(key)
                yield (line, key)
            keys.difference_update(source_keys)
        pb.update(gettext('Walking content'), total, total)

    def _make_line_delta(self, delta_seq, new_content):
        """Generate a line delta from delta_seq and new_content."""
        diff_hunks = []
        for op in delta_seq.get_opcodes():
            if op[0] == 'equal':
                continue
            diff_hunks.append((op[1], op[2], op[4] - op[3], new_content._lines[op[3]:op[4]]))
        return diff_hunks

    def _merge_annotations(self, content, parents, parent_texts={}, delta=None, annotated=None, left_matching_blocks=None):
        """Merge annotations for content and generate deltas.

        This is done by comparing the annotations based on changes to the text
        and generating a delta on the resulting full texts. If annotations are
        not being created then a simple delta is created.
        """
        if left_matching_blocks is not None:
            delta_seq = diff._PrematchedMatcher(left_matching_blocks)
        else:
            delta_seq = None
        if annotated:
            for parent_key in parents:
                merge_content = self._get_content(parent_key, parent_texts)
                if parent_key == parents[0] and delta_seq is not None:
                    seq = delta_seq
                else:
                    seq = patiencediff.PatienceSequenceMatcher(None, merge_content.text(), content.text())
                for i, j, n in seq.get_matching_blocks():
                    if n == 0:
                        continue
                    content._lines[j:j + n] = merge_content._lines[i:i + n]
            if content._lines and (not content._lines[-1][1].endswith(b'\n')):
                line = content._lines[-1][1] + b'\n'
                content._lines[-1] = (content._lines[-1][0], line)
        if delta:
            if delta_seq is None:
                reference_content = self._get_content(parents[0], parent_texts)
                new_texts = content.text()
                old_texts = reference_content.text()
                delta_seq = patiencediff.PatienceSequenceMatcher(None, old_texts, new_texts)
            return self._make_line_delta(delta_seq, content)

    def _parse_record(self, version_id, data):
        """Parse an original format knit record.

        These have the last element of the key only present in the stored data.
        """
        rec, record_contents = self._parse_record_unchecked(data)
        self._check_header_version(rec, version_id)
        return (record_contents, rec[3])

    def _parse_record_header(self, key, raw_data):
        """Parse a record header for consistency.

        :return: the header and the decompressor stream.
                 as (stream, header_record)
        """
        df = gzip.GzipFile(mode='rb', fileobj=BytesIO(raw_data))
        try:
            rec = self._check_header(key, df.readline())
        except Exception as e:
            raise KnitCorrupt(self, 'While reading {%s} got %s(%s)' % (key, e.__class__.__name__, str(e)))
        return (df, rec)

    def _parse_record_unchecked(self, data):
        with gzip.GzipFile(mode='rb', fileobj=BytesIO(data)) as df:
            try:
                record_contents = df.readlines()
            except Exception as e:
                raise KnitCorrupt(self, 'Corrupt compressed record %r, got %s(%s)' % (data, e.__class__.__name__, str(e)))
            header = record_contents.pop(0)
            rec = self._split_header(header)
            last_line = record_contents.pop()
            if len(record_contents) != int(rec[2]):
                raise KnitCorrupt(self, 'incorrect number of lines %s != %s for version {%s} %s' % (len(record_contents), int(rec[2]), rec[1], record_contents))
            if last_line != b'end %s\n' % rec[1]:
                raise KnitCorrupt(self, 'unexpected version end line %r, wanted %r' % (last_line, rec[1]))
        return (rec, record_contents)

    def _read_records_iter(self, records):
        """Read text records from data file and yield result.

        The result will be returned in whatever is the fastest to read.
        Not by the order requested. Also, multiple requests for the same
        record will only yield 1 response.

        :param records: A list of (key, access_memo) entries
        :return: Yields (key, contents, digest) in the order
                 read, not the order requested
        """
        if not records:
            return
        needed_records = sorted(set(records), key=operator.itemgetter(1))
        if not needed_records:
            return
        raw_data = self._access.get_raw_records([index_memo for key, index_memo in needed_records])
        for (key, index_memo), data in zip(needed_records, raw_data):
            content, digest = self._parse_record(key[-1], data)
            yield (key, content, digest)

    def _read_records_iter_raw(self, records):
        """Read text records from data file and yield raw data.

        This unpacks enough of the text record to validate the id is
        as expected but thats all.

        Each item the iterator yields is (key, bytes,
            expected_sha1_of_full_text).
        """
        for key, data in self._read_records_iter_unchecked(records):
            df, rec = self._parse_record_header(key, data)
            df.close()
            yield (key, data, rec[3])

    def _read_records_iter_unchecked(self, records):
        """Read text records from data file and yield raw data.

        No validation is done.

        Yields tuples of (key, data).
        """
        if len(records):
            needed_offsets = [index_memo for key, index_memo in records]
            raw_records = self._access.get_raw_records(needed_offsets)
        for key, index_memo in records:
            data = next(raw_records)
            yield (key, data)

    def _record_to_data(self, key, digest, lines, dense_lines=None):
        """Convert key, digest, lines into a raw data block.

        :param key: The key of the record. Currently keys are always serialised
            using just the trailing component.
        :param dense_lines: The bytes of lines but in a denser form. For
            instance, if lines is a list of 1000 bytestrings each ending in
            \\n, dense_lines may be a list with one line in it, containing all
            the 1000's lines and their \\n's. Using dense_lines if it is
            already known is a win because the string join to create bytes in
            this function spends less time resizing the final string.
        :return: (len, chunked bytestring with compressed data)
        """
        chunks = [b'version %s %d %s\n' % (key[-1], len(lines), digest)]
        chunks.extend(dense_lines or lines)
        chunks.append(b'end ' + key[-1] + b'\n')
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise AssertionError('data must be plain bytes was %s' % type(chunk))
        if lines and (not lines[-1].endswith(b'\n')):
            raise ValueError('corrupt lines value %r' % lines)
        compressed_chunks = tuned_gzip.chunks_to_gzip(chunks)
        return (sum(map(len, compressed_chunks)), compressed_chunks)

    def _split_header(self, line):
        rec = line.split()
        if len(rec) != 4:
            raise KnitCorrupt(self, 'unexpected number of elements in record header')
        return rec

    def keys(self):
        """See VersionedFiles.keys."""
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(2, 'keys scales with size of history')
        sources = [self._index] + self._immediate_fallback_vfs
        result = set()
        for source in sources:
            result.update(source.keys())
        return result