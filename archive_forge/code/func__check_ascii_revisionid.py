from typing import List, Type, TYPE_CHECKING, Optional, Iterable
from .lazy_import import lazy_import
import time
from breezy import (
from breezy.i18n import gettext
from . import controldir, debug, errors, graph, registry, revision as _mod_revision, ui
from .decorators import only_raises
from .inter import InterObject
from .lock import LogicalLockResult, _RelockDebugMixin
from .revisiontree import RevisionTree
from .trace import (log_exception_quietly, mutter, mutter_callsite, note,
def _check_ascii_revisionid(self, revision_id, method):
    """Private helper for ascii-only repositories."""
    if revision_id is not None:
        if isinstance(revision_id, str):
            try:
                revision_id.encode('ascii')
            except UnicodeEncodeError:
                raise errors.NonAsciiRevisionId(method, self)
        else:
            try:
                revision_id.decode('ascii')
            except UnicodeDecodeError:
                raise errors.NonAsciiRevisionId(method, self)