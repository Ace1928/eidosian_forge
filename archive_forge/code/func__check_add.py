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