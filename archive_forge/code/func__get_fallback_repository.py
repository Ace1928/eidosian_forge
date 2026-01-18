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
def _get_fallback_repository(self, url, possible_transports):
    """Get the repository we fallback to at url."""
    url = urlutils.join(self.base, url)
    a_branch = Branch.open(url, possible_transports=possible_transports)
    return a_branch.repository