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
def _get_needed_texts(self, key, pb=None):
    if len(self._vf._immediate_fallback_vfs) > 0:
        yield from annotate.Annotator._get_needed_texts(self, key, pb=pb)
        return
    while True:
        try:
            records, ann_keys = self._get_build_graph(key)
            for idx, (sub_key, text, num_lines) in enumerate(self._extract_texts(records)):
                if pb is not None:
                    pb.update(gettext('annotating'), idx, len(records))
                yield (sub_key, text, num_lines)
            for sub_key in ann_keys:
                text = self._text_cache[sub_key]
                num_lines = len(text)
                yield (sub_key, text, num_lines)
            return
        except pack_repo.RetryWithNewPacks as e:
            self._vf._access.reload_or_raise(e)
            self._all_build_details.clear()