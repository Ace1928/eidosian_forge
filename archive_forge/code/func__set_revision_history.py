from .. import debug, errors
from .. import revision as _mod_revision
from ..branch import Branch
from ..trace import mutter_callsite
from .branch import BranchFormatMetadir, BzrBranch
def _set_revision_history(self, rev_history):
    if 'evil' in debug.debug_flags:
        mutter_callsite(3, 'set_revision_history scales with history.')
    check_not_reserved_id = _mod_revision.check_not_reserved_id
    for rev_id in rev_history:
        check_not_reserved_id(rev_id)
    if Branch.hooks['post_change_branch_tip']:
        old_revno, old_revid = self.last_revision_info()
    if len(rev_history) == 0:
        revid = _mod_revision.NULL_REVISION
    else:
        revid = rev_history[-1]
    self._run_pre_change_branch_tip_hooks(len(rev_history), revid)
    self._write_revision_history(rev_history)
    self._clear_cached_state()
    self._cache_revision_history(rev_history)
    if Branch.hooks['post_change_branch_tip']:
        self._run_post_change_branch_tip_hooks(old_revno, old_revid)