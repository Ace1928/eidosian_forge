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
class _ContentMapGenerator:
    """Generate texts or expose raw deltas for a set of texts."""

    def __init__(self, ordering='unordered'):
        self._ordering = ordering

    def _get_content(self, key):
        """Get the content object for key."""
        if key in self.nonlocal_keys:
            record = next(self.get_record_stream())
            lines = record.get_bytes_as('lines')
            return PlainKnitContent(lines, record.key)
        else:
            return self._get_one_work(key)

    def get_record_stream(self):
        """Get a record stream for the keys requested during __init__."""
        yield from self._work()

    def _work(self):
        """Produce maps of text and KnitContents as dicts.

        :return: (text_map, content_map) where text_map contains the texts for
            the requested versions and content_map contains the KnitContents.
        """
        if self.global_map is None:
            self.global_map = self.vf.get_parent_map(self.keys)
        nonlocal_keys = self.nonlocal_keys
        missing_keys = set(nonlocal_keys)
        for source in self.vf._immediate_fallback_vfs:
            if not missing_keys:
                break
            for record in source.get_record_stream(missing_keys, self._ordering, True):
                if record.storage_kind == 'absent':
                    continue
                missing_keys.remove(record.key)
                yield record
        if self._raw_record_map is None:
            raise AssertionError('_raw_record_map should have been filled')
        first = True
        for key in self.keys:
            if key in self.nonlocal_keys:
                continue
            yield LazyKnitContentFactory(key, self.global_map[key], self, first)
            first = False

    def _get_one_work(self, requested_key):
        if requested_key in self._contents_map:
            return self._contents_map[requested_key]
        multiple_versions = len(self.keys) != 1
        if self._record_map is None:
            self._record_map = self.vf._raw_map_to_record_map(self._raw_record_map)
        record_map = self._record_map
        for key in self.keys:
            if key in self.nonlocal_keys:
                continue
            components = []
            cursor = key
            while cursor is not None:
                try:
                    record, record_details, digest, next = record_map[cursor]
                except KeyError:
                    raise RevisionNotPresent(cursor, self)
                components.append((cursor, record, record_details, digest))
                cursor = next
                if cursor in self._contents_map:
                    components.append((cursor, None, None, None))
                    break
            content = None
            for component_id, record, record_details, digest in reversed(components):
                if component_id in self._contents_map:
                    content = self._contents_map[component_id]
                else:
                    content, delta = self._factory.parse_record(key[-1], record, record_details, content, copy_base_content=multiple_versions)
                    if multiple_versions:
                        self._contents_map[component_id] = content
            text = content.text()
            actual_sha = sha_strings(text)
            if actual_sha != digest:
                raise SHA1KnitCorrupt(self, actual_sha, digest, key, text)
        if multiple_versions:
            return self._contents_map[requested_key]
        else:
            return content

    def _wire_bytes(self):
        """Get the bytes to put on the wire for 'key'.

        The first collection of bytes asked for returns the serialised
        raw_record_map and the additional details (key, parent) for key.
        Subsequent calls return just the additional details (key, parent).
        The wire storage_kind given for the first key is 'knit-delta-closure',
        For subsequent keys it is 'knit-delta-closure-ref'.

        :param key: A key from the content generator.
        :return: Bytes to put on the wire.
        """
        lines = []
        lines.append(b'knit-delta-closure')
        if self.vf._factory.annotated:
            lines.append(b'annotated')
        else:
            lines.append(b'')
        lines.append(b'\t'.join((b'\x00'.join(key) for key in self.keys if key not in self.nonlocal_keys)))
        map_byte_list = []
        for key, (record_bytes, (method, noeol), next) in self._raw_record_map.items():
            key_bytes = b'\x00'.join(key)
            parents = self.global_map.get(key, None)
            if parents is None:
                parent_bytes = b'None:'
            else:
                parent_bytes = b'\t'.join((b'\x00'.join(key) for key in parents))
            method_bytes = method.encode('ascii')
            if noeol:
                noeol_bytes = b'T'
            else:
                noeol_bytes = b'F'
            if next:
                next_bytes = b'\x00'.join(next)
            else:
                next_bytes = b''
            map_byte_list.append(b'\n'.join([key_bytes, parent_bytes, method_bytes, noeol_bytes, next_bytes, b'%d' % len(record_bytes), record_bytes]))
        map_bytes = b''.join(map_byte_list)
        lines.append(map_bytes)
        bytes = b'\n'.join(lines)
        return bytes