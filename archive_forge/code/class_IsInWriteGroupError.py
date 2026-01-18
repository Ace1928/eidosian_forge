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
class IsInWriteGroupError(errors.InternalBzrError):
    _fmt = 'May not refresh_data of repo %(repo)s while in a write group.'

    def __init__(self, repo):
        errors.InternalBzrError.__init__(self, repo=repo)