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
def _do_dotted_revno_to_revision_id(self, revno):
    """Worker function for dotted_revno_to_revision_id.

        Subclasses should override this if they wish to
        provide a more efficient implementation.
        """
    if len(revno) == 1:
        try:
            return self.get_rev_id(revno[0])
        except errors.RevisionNotPresent as exc:
            raise errors.GhostRevisionsHaveNoRevno(revno[0], exc.revision_id) from exc
    revision_id_to_revno = self.get_revision_id_to_revno_map()
    revision_ids = [revision_id for revision_id, this_revno in revision_id_to_revno.items() if revno == this_revno]
    if len(revision_ids) == 1:
        return revision_ids[0]
    else:
        revno_str = '.'.join(map(str, revno))
        raise errors.NoSuchRevision(self, revno_str)