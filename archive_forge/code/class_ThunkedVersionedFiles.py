import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
class ThunkedVersionedFiles(VersionedFiles):
    """Storage for many versioned files thunked onto a 'VersionedFile' class.

    This object allows a single keyspace for accessing the history graph and
    contents of named bytestrings.

    Currently no implementation allows the graph of different key prefixes to
    intersect, but the API does allow such implementations in the future.
    """

    def __init__(self, transport, file_factory, mapper, is_locked):
        """Create a ThunkedVersionedFiles."""
        self._transport = transport
        self._file_factory = file_factory
        self._mapper = mapper
        self._is_locked = is_locked

    def add_content(self, factory, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False):
        """See VersionedFiles.add_content()."""
        lines = factory.get_bytes_as('lines')
        return self.add_lines(factory.key, factory.parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=True)

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        """See VersionedFiles.add_lines()."""
        path = self._mapper.map(key)
        version_id = key[-1]
        parents = [parent[-1] for parent in parents]
        vf = self._get_vf(path)
        try:
            try:
                return vf.add_lines_with_ghosts(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)
            except NotImplementedError:
                return vf.add_lines(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)
        except _mod_transport.NoSuchFile:
            self._transport.mkdir(osutils.dirname(path))
            try:
                return vf.add_lines_with_ghosts(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)
            except NotImplementedError:
                return vf.add_lines(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)

    def annotate(self, key):
        """Return a list of (version-key, line) tuples for the text of key.

        :raise RevisionNotPresent: If the key is not present.
        """
        prefix = key[:-1]
        path = self._mapper.map(prefix)
        vf = self._get_vf(path)
        origins = vf.annotate(key[-1])
        result = []
        for origin, line in origins:
            result.append((prefix + (origin,), line))
        return result

    def check(self, progress_bar=None, keys=None):
        """See VersionedFiles.check()."""
        for prefix, vf in self._iter_all_components():
            vf.check()
        if keys is not None:
            return self.get_record_stream(keys, 'unordered', True)

    def get_parent_map(self, keys):
        """Get a map of the parents of keys.

        :param keys: The keys to look up parents for.
        :return: A mapping from keys to parents. Absent keys are absent from
            the mapping.
        """
        prefixes = self._partition_keys(keys)
        result = {}
        for prefix, suffixes in prefixes.items():
            path = self._mapper.map(prefix)
            vf = self._get_vf(path)
            parent_map = vf.get_parent_map(suffixes)
            for key, parents in parent_map.items():
                result[prefix + (key,)] = tuple((prefix + (parent,) for parent in parents))
        return result

    def _get_vf(self, path):
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)
        return self._file_factory(path, self._transport, create=True, get_scope=lambda: None)

    def _partition_keys(self, keys):
        """Turn keys into a dict of prefix:suffix_list."""
        result = {}
        for key in keys:
            prefix_keys = result.setdefault(key[:-1], [])
            prefix_keys.append(key[-1])
        return result

    def _iter_all_prefixes(self):
        if isinstance(self._mapper, ConstantMapper):
            paths = [self._mapper.map(())]
            prefixes = [()]
        else:
            relpaths = set()
            for quoted_relpath in self._transport.iter_files_recursive():
                path, ext = os.path.splitext(quoted_relpath)
                relpaths.add(path)
            paths = list(relpaths)
            prefixes = [self._mapper.unmap(path) for path in paths]
        return zip(paths, prefixes)

    def get_record_stream(self, keys, ordering, include_delta_closure):
        """See VersionedFiles.get_record_stream()."""
        keys = sorted(keys)
        for prefix, suffixes, vf in self._iter_keys_vf(keys):
            suffixes = [(suffix,) for suffix in suffixes]
            for record in vf.get_record_stream(suffixes, ordering, include_delta_closure):
                if record.parents is not None:
                    record.parents = tuple((prefix + parent for parent in record.parents))
                record.key = prefix + record.key
                yield record

    def _iter_keys_vf(self, keys):
        prefixes = self._partition_keys(keys)
        sha1s = {}
        for prefix, suffixes in prefixes.items():
            path = self._mapper.map(prefix)
            vf = self._get_vf(path)
            yield (prefix, suffixes, vf)

    def get_sha1s(self, keys):
        """See VersionedFiles.get_sha1s()."""
        sha1s = {}
        for prefix, suffixes, vf in self._iter_keys_vf(keys):
            vf_sha1s = vf.get_sha1s(suffixes)
            for suffix, sha1 in vf_sha1s.items():
                sha1s[prefix + (suffix,)] = sha1
        return sha1s

    def insert_record_stream(self, stream):
        """Insert a record stream into this container.

        :param stream: A stream of records to insert.
        :return: None
        :seealso VersionedFile.get_record_stream:
        """
        for record in stream:
            prefix = record.key[:-1]
            key = record.key[-1:]
            if record.parents is not None:
                parents = [parent[-1:] for parent in record.parents]
            else:
                parents = None
            thunk_record = AdapterFactory(key, parents, record)
            path = self._mapper.map(prefix)
            vf = self._get_vf(path)
            vf.insert_record_stream([thunk_record])

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
        for prefix, suffixes, vf in self._iter_keys_vf(keys):
            for line, version in vf.iter_lines_added_or_present_in_versions(suffixes):
                yield (line, prefix + (version,))

    def _iter_all_components(self):
        for path, prefix in self._iter_all_prefixes():
            yield (prefix, self._get_vf(path))

    def keys(self):
        """See VersionedFiles.keys()."""
        result = set()
        for prefix, vf in self._iter_all_components():
            for suffix in vf.versions():
                result.add(prefix + (suffix,))
        return result