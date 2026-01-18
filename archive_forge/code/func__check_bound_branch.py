from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _check_bound_branch(self, stack, possible_master_transports=None):
    """Check to see if the local branch is bound.

        If it is bound, then most of the commit will actually be
        done using the remote branch as the target branch.
        Only at the end will the local branch be updated.
        """
    if self.local and (not self.branch.get_bound_location()):
        raise errors.LocalRequiresBoundBranch()
    if not self.local:
        self.master_branch = self.branch.get_master_branch(possible_master_transports)
    if not self.master_branch:
        self.master_branch = self.branch
        return
    master_bound_location = self.master_branch.get_bound_location()
    if master_bound_location:
        raise errors.CommitToDoubleBoundBranch(self.branch, self.master_branch, master_bound_location)
    master_revid = self.master_branch.last_revision()
    local_revid = self.branch.last_revision()
    if local_revid != master_revid:
        raise errors.BoundBranchOutOfDate(self.branch, self.master_branch)
    self.bound_branch = self.branch
    stack.enter_context(self.master_branch.lock_write())