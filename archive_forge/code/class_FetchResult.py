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
class FetchResult:
    """Result of a fetch operation.

    Attributes:
      revidmap: For lossy fetches, map from source revid to target revid.
      total_fetched: Number of revisions fetched
    """

    def __init__(self, total_fetched=None, revidmap=None):
        self.total_fetched = total_fetched
        self.revidmap = revidmap