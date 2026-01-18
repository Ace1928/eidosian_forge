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