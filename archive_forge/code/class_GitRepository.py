from io import BytesIO
from dulwich.errors import NotCommitError
from dulwich.object_store import peel_sha, tree_lookup_path
from dulwich.objects import ZERO_SHA, Commit
from .. import check, errors
from .. import graph as _mod_graph
from .. import lock, repository
from .. import revision as _mod_revision
from .. import trace, transactions, ui
from ..decorators import only_raises
from ..foreign import ForeignRepository
from .filegraph import GitFileLastChangeScanner, GitFileParentProvider
from .mapping import (default_mapping, encode_git_path, foreign_vcs_git,
from .tree import GitRevisionTree
class GitRepository(ForeignRepository):
    """An adapter to git repositories for bzr."""
    vcs = foreign_vcs_git
    chk_bytes = None

    def __init__(self, gitdir):
        self._transport = gitdir.root_transport
        super().__init__(GitRepositoryFormat(), gitdir, control_files=None)
        self.base = gitdir.root_transport.base
        self._lock_mode = None
        self._lock_count = 0

    def add_fallback_repository(self, basis_url):
        raise errors.UnstackableRepositoryFormat(self._format, self.control_transport.base)

    def is_shared(self):
        return False

    def get_physical_lock_status(self):
        return False

    def lock_write(self):
        """See Branch.lock_write()."""
        if self._lock_mode:
            if self._lock_mode != 'w':
                raise errors.ReadOnlyError(self)
            self._lock_count += 1
        else:
            self._lock_mode = 'w'
            self._lock_count = 1
            self._transaction = transactions.WriteTransaction()
        return repository.RepositoryWriteLockResult(self.unlock, None)

    def break_lock(self):
        raise NotImplementedError(self.break_lock)

    def dont_leave_lock_in_place(self):
        raise NotImplementedError(self.dont_leave_lock_in_place)

    def leave_lock_in_place(self):
        raise NotImplementedError(self.leave_lock_in_place)

    def lock_read(self):
        if self._lock_mode:
            if self._lock_mode not in ('r', 'w'):
                raise AssertionError
            self._lock_count += 1
        else:
            self._lock_mode = 'r'
            self._lock_count = 1
            self._transaction = transactions.ReadOnlyTransaction()
        return lock.LogicalLockResult(self.unlock)

    @only_raises(errors.LockNotHeld, errors.LockBroken)
    def unlock(self):
        if self._lock_count == 0:
            raise errors.LockNotHeld(self)
        if self._lock_count == 1 and self._lock_mode == 'w':
            if self._write_group is not None:
                self.abort_write_group()
                self._lock_count -= 1
                self._lock_mode = None
                raise errors.BzrError('Must end write groups before releasing write locks.')
        self._lock_count -= 1
        if self._lock_count == 0:
            self._lock_mode = None
            transaction = self._transaction
            self._transaction = None
            transaction.finish()
            if hasattr(self, '_git'):
                self._git.close()

    def is_write_locked(self):
        return self._lock_mode == 'w'

    def is_locked(self):
        return self._lock_mode is not None

    def get_transaction(self):
        """See Repository.get_transaction()."""
        if self._transaction is None:
            return transactions.PassThroughTransaction()
        else:
            return self._transaction

    def reconcile(self, other=None, thorough=False):
        """Reconcile this repository."""
        from ..reconcile import ReconcileResult
        ret = ReconcileResult()
        ret.aborted = False
        return ret

    def supports_rich_root(self):
        return True

    def get_mapping(self):
        return default_mapping

    def make_working_trees(self):
        raise NotImplementedError(self.make_working_trees)

    def revision_graph_can_have_wrong_parents(self):
        return False

    def add_signature_text(self, revid, signature):
        raise errors.UnsupportedOperation(self.add_signature_text, self)

    def sign_revision(self, revision_id, gpg_strategy):
        raise errors.UnsupportedOperation(self.add_signature_text, self)