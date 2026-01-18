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
class BranchWriteLockResult(LogicalLockResult):
    """The result of write locking a branch.

    Attributes:
      token: The token obtained from the underlying branch lock, or
        None.
      unlock: A callable which will unlock the lock.
    """

    def __repr__(self):
        return 'BranchWriteLockResult({!r}, {!r})'.format(self.unlock, self.token)