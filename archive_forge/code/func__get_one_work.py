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