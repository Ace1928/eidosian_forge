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
def _extract_texts(self, records):
    """Extract the various texts needed based on records"""
    pending_deltas = {}
    for key, record, digest in self._vf._read_records_iter(records):
        details = self._all_build_details[key]
        _, compression_parent, parent_keys, record_details = details
        lines = self._expand_record(key, parent_keys, compression_parent, record, record_details)
        if lines is None:
            continue
        yield_this_text = self._check_ready_for_annotations(key, parent_keys)
        if yield_this_text:
            yield (key, lines, len(lines))
        to_process = self._process_pending(key)
        while to_process:
            this_process = to_process
            to_process = []
            for key in this_process:
                lines = self._text_cache[key]
                yield (key, lines, len(lines))
                to_process.extend(self._process_pending(key))