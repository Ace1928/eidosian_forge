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
def _gen_revno_map(self):
    """Create a new mapping from revision ids to dotted revnos.

        Dotted revnos are generated based on the current tip in the revision
        history.
        This is the worker function for get_revision_id_to_revno_map, which
        just caches the return value.

        Returns: A dictionary mapping revision_id => dotted revno.
        """
    revision_id_to_revno = {rev_id: revno for rev_id, depth, revno, end_of_merge in self.iter_merge_sorted_revisions()}
    return revision_id_to_revno