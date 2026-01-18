from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def _cache_revision_history(self, rev_history):
    """Set the cached revision history to rev_history.

        The revision_history method will use this cache to avoid regenerating
        the revision history.

        This API is semi-public; it only for use by subclasses, all other code
        should consider it to be private.
        """
    self._revision_history_cache = rev_history