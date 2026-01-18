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