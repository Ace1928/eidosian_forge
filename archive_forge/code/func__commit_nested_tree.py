from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _commit_nested_tree(self, path):
    """Commit a nested tree."""
    sub_tree = self.work_tree.get_nested_tree(path)
    if sub_tree.branch.repository.has_same_location(self.work_tree.branch.repository):
        sub_tree.branch.repository = self.work_tree.branch.repository
    try:
        return sub_tree.commit(message=None, revprops=self.revprops, recursive=self.recursive, message_callback=self.message_callback, timestamp=self.timestamp, timezone=self.timezone, committer=self.committer, allow_pointless=self.allow_pointless, strict=self.strict, verbose=self.verbose, local=self.local, reporter=self.reporter)
    except PointlessCommit:
        return self.work_tree.get_reference_revision(path)