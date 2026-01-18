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
@staticmethod
def _same_model(source, target):
    """True if source and target have the same data representation.

        Note: this is always called on the base class; overriding it in a
        subclass will have no effect.
        """
    try:
        InterRepository._assert_same_model(source, target)
        return True
    except errors.IncompatibleRepositories as e:
        return False