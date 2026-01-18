from contextlib import ExitStack
import breezy.config
from . import debug, errors, trace, ui
from .branch import Branch
from .errors import BzrError, ConflictsInTree, StrictCommitFailed
from .i18n import gettext
from .osutils import (get_user_encoding, is_inside_any, minimum_path_selection,
from .trace import is_quiet, mutter, note
from .urlutils import unescape_for_display
def _process_hooks(self, hook_name, old_revno, new_revno):
    if not Branch.hooks[hook_name]:
        return
    if not self.bound_branch:
        hook_master = self.branch
        hook_local = None
    else:
        hook_master = self.master_branch
        hook_local = self.branch
    if self.parents:
        old_revid = self.parents[0]
    else:
        old_revid = breezy.revision.NULL_REVISION
    if hook_name == 'pre_commit':
        future_tree = self.builder.revision_tree()
        tree_delta = future_tree.changes_from(self.basis_tree, include_root=True)
    for hook in Branch.hooks[hook_name]:
        self.pb_stage_name = 'Running %s hooks [%s]' % (hook_name, Branch.hooks.get_hook_name(hook))
        self._emit_progress()
        if 'hooks' in debug.debug_flags:
            mutter('Invoking commit hook: %r', hook)
        if hook_name == 'post_commit':
            hook(hook_local, hook_master, old_revno, old_revid, new_revno, self.rev_id)
        elif hook_name == 'pre_commit':
            hook(hook_local, hook_master, old_revno, old_revid, new_revno, self.rev_id, tree_delta, future_tree)