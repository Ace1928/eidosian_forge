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
def _revision_history(self) -> List[RevisionID]:
    if 'evil' in debug.debug_flags:
        mutter_callsite(3, 'revision_history scales with history.')
    if self._revision_history_cache is not None:
        history = self._revision_history_cache
    else:
        history = self._gen_revision_history()
        self._cache_revision_history(history)
    return list(history)