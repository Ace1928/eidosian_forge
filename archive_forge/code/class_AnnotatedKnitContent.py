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
class AnnotatedKnitContent(KnitContent):
    """Annotated content."""

    def __init__(self, lines):
        KnitContent.__init__(self)
        self._lines = list(lines)

    def annotate(self):
        """Return a list of (origin, text) for each content line."""
        lines = self._lines[:]
        if self._should_strip_eol:
            origin, last_line = lines[-1]
            lines[-1] = (origin, last_line.rstrip(b'\n'))
        return lines

    def apply_delta(self, delta, new_version_id):
        """Apply delta to this object to become new_version_id."""
        offset = 0
        lines = self._lines
        for start, end, count, delta_lines in delta:
            lines[offset + start:offset + end] = delta_lines
            offset = offset + (start - end) + count

    def text(self):
        try:
            lines = [text for origin, text in self._lines]
        except ValueError as e:
            raise KnitCorrupt(self, 'line in annotated knit missing annotation information: %s' % (e,))
        if self._should_strip_eol:
            lines[-1] = lines[-1].rstrip(b'\n')
        return lines

    def copy(self):
        return AnnotatedKnitContent(self._lines)