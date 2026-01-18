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
def _make_line_delta(self, delta_seq, new_content):
    """Generate a line delta from delta_seq and new_content."""
    diff_hunks = []
    for op in delta_seq.get_opcodes():
        if op[0] == 'equal':
            continue
        diff_hunks.append((op[1], op[2], op[4] - op[3], new_content._lines[op[3]:op[4]]))
    return diff_hunks