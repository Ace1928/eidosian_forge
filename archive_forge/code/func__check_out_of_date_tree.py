from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _check_out_of_date_tree(self):
    """Check that the working tree is up to date.

        :return: old_revision_number, old_revision_id, new_revision_number
            tuple
        """
    try:
        first_tree_parent = self.work_tree.get_parent_ids()[0]
    except IndexError:
        first_tree_parent = breezy.revision.NULL_REVISION
    if self.master_branch._format.stores_revno() or self.config_stack.get('calculate_revnos'):
        try:
            old_revno, master_last = self.master_branch.last_revision_info()
        except errors.UnsupportedOperation:
            master_last = self.master_branch.last_revision()
            old_revno = self.branch.revision_id_to_revno(master_last)
    else:
        master_last = self.master_branch.last_revision()
        old_revno = None
    if master_last != first_tree_parent:
        if master_last != breezy.revision.NULL_REVISION:
            raise errors.OutOfDateTree(self.work_tree)
    if old_revno is not None and self.branch.repository.has_revision(first_tree_parent):
        new_revno = old_revno + 1
    else:
        new_revno = None
    return (old_revno, master_last, new_revno)