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
def _expand_record(self, key, parent_keys, compression_parent, record, record_details):
    delta = None
    if compression_parent:
        if compression_parent not in self._content_objects:
            self._pending_deltas.setdefault(compression_parent, []).append((key, parent_keys, record, record_details))
            return None
        num = self._num_compression_children[compression_parent]
        num -= 1
        if num == 0:
            base_content = self._content_objects.pop(compression_parent)
            self._num_compression_children.pop(compression_parent)
        else:
            self._num_compression_children[compression_parent] = num
            base_content = self._content_objects[compression_parent]
        content, delta = self._vf._factory.parse_record(key, record, record_details, base_content, copy_base_content=True)
    else:
        content, _ = self._vf._factory.parse_record(key, record, record_details, None)
    if self._num_compression_children.get(key, 0) > 0:
        self._content_objects[key] = content
    lines = content.text()
    self._text_cache[key] = lines
    if delta is not None:
        self._cache_delta_blocks(key, compression_parent, delta, lines)
    return lines