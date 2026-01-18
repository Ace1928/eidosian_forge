from .. import debug, errors
from .. import revision as _mod_revision
from ..branch import Branch
from ..trace import mutter_callsite
from .branch import BranchFormatMetadir, BzrBranch
class FullHistoryBzrBranch(BzrBranch):
    """Bzr branch which contains the full revision history."""

    def set_last_revision_info(self, revno, revision_id):
        if not revision_id or not isinstance(revision_id, bytes):
            raise errors.InvalidRevisionId(revision_id=revision_id, branch=self)
        with self.lock_write():
            history = self._lefthand_history(revision_id)
            if len(history) != revno:
                raise AssertionError('%d != %d' % (len(history), revno))
            self._set_revision_history(history)

    def _read_last_revision_info(self):
        rh = self._revision_history()
        revno = len(rh)
        if revno:
            return (revno, rh[-1])
        else:
            return (0, _mod_revision.NULL_REVISION)

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

    def _write_revision_history(self, history):
        """Factored out of set_revision_history.

        This performs the actual writing to disk.
        It is intended to be called by set_revision_history."""
        self._transport.put_bytes('revision-history', b'\n'.join(history), mode=self.controldir._get_file_mode())

    def _gen_revision_history(self):
        history = self._transport.get_bytes('revision-history').split(b'\n')
        if history[-1:] == [b'']:
            history.pop()
        return history

    def _synchronize_history(self, destination, revision_id):
        if not isinstance(destination, FullHistoryBzrBranch):
            super(BzrBranch, self)._synchronize_history(destination, revision_id)
            return
        if revision_id == _mod_revision.NULL_REVISION:
            new_history = []
        else:
            new_history = self._revision_history()
        if revision_id is not None and new_history != []:
            try:
                new_history = new_history[:new_history.index(revision_id) + 1]
            except ValueError:
                rev = self.repository.get_revision(revision_id)
                new_history = rev.get_history(self.repository)[1:]
        destination._set_revision_history(new_history)

    def generate_revision_history(self, revision_id, last_rev=None, other_branch=None):
        """Create a new revision history that will finish with revision_id.

        :param revision_id: the new tip to use.
        :param last_rev: The previous last_revision. If not None, then this
            must be a ancestory of revision_id, or DivergedBranches is raised.
        :param other_branch: The other branch that DivergedBranches should
            raise with respect to.
        """
        with self.lock_write():
            self._set_revision_history(self._lefthand_history(revision_id, last_rev, other_branch))