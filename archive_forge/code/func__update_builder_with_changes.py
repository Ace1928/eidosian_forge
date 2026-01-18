from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _update_builder_with_changes(self):
    """Update the commit builder with the data about what has changed.
        """
    specific_files = self.specific_files
    mutter('Selecting files for commit with filter %r', specific_files)
    self._check_strict()
    iter_changes = self.work_tree.iter_changes(self.basis_tree, specific_files=specific_files)
    if self.exclude:
        iter_changes = filter_excluded(iter_changes, self.exclude)
    iter_changes = self._filter_iter_changes(iter_changes)
    for path, fs_hash in self.builder.record_iter_changes(self.work_tree, self.basis_revid, iter_changes):
        self.work_tree._observed_sha1(path, fs_hash)