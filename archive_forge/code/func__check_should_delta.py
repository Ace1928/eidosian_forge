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
def _check_should_delta(self, parent):
    """Iterate back through the parent listing, looking for a fulltext.

        This is used when we want to decide whether to add a delta or a new
        fulltext. It searches for _max_delta_chain parents. When it finds a
        fulltext parent, it sees if the total size of the deltas leading up to
        it is large enough to indicate that we want a new full text anyway.

        Return True if we should create a new delta, False if we should use a
        full text.
        """
    delta_size = 0
    fulltext_size = None
    for count in range(self._max_delta_chain):
        try:
            build_details = self._index.get_build_details([parent])
            parent_details = build_details[parent]
        except (RevisionNotPresent, KeyError) as e:
            return False
        index_memo, compression_parent, _, _ = parent_details
        _, _, size = index_memo
        if compression_parent is None:
            fulltext_size = size
            break
        delta_size += size
        parent = compression_parent
    else:
        return False
    return fulltext_size > delta_size