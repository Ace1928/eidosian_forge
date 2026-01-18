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
class ChangeBranchTipParams:
    """Object holding parameters passed to `*_change_branch_tip` hooks.

    There are 5 fields that hooks may wish to access:

    Attributes:
      branch: the branch being changed
      old_revno: revision number before the change
      new_revno: revision number after the change
      old_revid: revision id before the change
      new_revid: revision id after the change

    The revid fields are strings. The revno fields are integers.
    """

    def __init__(self, branch, old_revno, new_revno, old_revid, new_revid):
        """Create a group of ChangeBranchTip parameters.

        Args:
          branch: The branch being changed.
          old_revno: Revision number before the change.
          new_revno: Revision number after the change.
          old_revid: Tip revision id before the change.
          new_revid: Tip revision id after the change.
        """
        self.branch = branch
        self.old_revno = old_revno
        self.new_revno = new_revno
        self.old_revid = old_revid
        self.new_revid = new_revid

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return '<{} of {} from ({}, {}) to ({}, {})>'.format(self.__class__.__name__, self.branch, self.old_revno, self.old_revid, self.new_revno, self.new_revid)