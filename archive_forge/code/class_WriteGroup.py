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
class WriteGroup:
    """Context manager that manages a write group.

    Raising an exception will result in the write group being aborted.
    """

    def __init__(self, repository, suppress_errors=False):
        self.repository = repository
        self._suppress_errors = suppress_errors

    def __enter__(self):
        self.repository.start_write_group()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.repository.abort_write_group(self._suppress_errors)
            return False
        else:
            self.repository.commit_write_group()