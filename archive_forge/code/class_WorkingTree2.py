from io import BytesIO
from ... import conflicts as _mod_conflicts
from ... import errors, lock, osutils
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...bzr import conflicts as _mod_bzr_conflicts
from ...bzr import inventory
from ...bzr import transform as bzr_transform
from ...bzr import xml5
from ...bzr.workingtree_3 import PreDirStateWorkingTree
from ...mutabletree import MutableTree
from ...transport.local import LocalTransport
from ...workingtree import WorkingTreeFormat
class WorkingTree2(PreDirStateWorkingTree):
    """This is the Format 2 working tree.

    This was the first weave based working tree.
     - uses os locks for locking.
     - uses the branch last-revision.
    """

    def __init__(self, basedir, *args, **kwargs):
        super().__init__(basedir, *args, **kwargs)
        if self._inventory is None:
            self.read_working_inventory()

    def _get_check_refs(self):
        """Return the references needed to perform a check of this tree."""
        return [('trees', self.last_revision())]

    def lock_tree_write(self):
        """See WorkingTree.lock_tree_write().

        In Format2 WorkingTrees we have a single lock for the branch and tree
        so lock_tree_write() degrades to lock_write().

        :return: An object with an unlock method which will release the lock
            obtained.
        """
        self.branch.lock_write()
        try:
            token = self._control_files.lock_write()
            return lock.LogicalLockResult(self.unlock, token)
        except:
            self.branch.unlock()
            raise

    def unlock(self):
        if self._control_files._lock_count == 3:
            self._cleanup()
            if self._inventory_is_modified:
                self.flush()
            self._write_hashcache_if_dirty()
        try:
            return self._control_files.unlock()
        finally:
            self.branch.unlock()

    def _iter_conflicts(self):
        conflicted = set()
        for path, file_class, file_kind, entry in self.list_files():
            stem = get_conflicted_stem(path)
            if stem is None:
                continue
            if stem not in conflicted:
                conflicted.add(stem)
                yield stem

    def conflicts(self):
        with self.lock_read():
            conflicts = _mod_conflicts.ConflictList()
            for conflicted in self._iter_conflicts():
                text = True
                try:
                    if osutils.file_kind(self.abspath(conflicted)) != 'file':
                        text = False
                except _mod_transport.NoSuchFile:
                    text = False
                if text is True:
                    for suffix in ('.THIS', '.OTHER'):
                        try:
                            kind = osutils.file_kind(self.abspath(conflicted + suffix))
                            if kind != 'file':
                                text = False
                        except _mod_transport.NoSuchFile:
                            text = False
                        if text is False:
                            break
                ctype = {True: 'text conflict', False: 'contents conflict'}[text]
                conflicts.append(_mod_bzr_conflicts.Conflict.factory(ctype, path=conflicted, file_id=self.path2id(conflicted)))
            return conflicts

    def set_conflicts(self, arg):
        raise errors.UnsupportedOperation(self.set_conflicts, self)

    def add_conflicts(self, arg):
        raise errors.UnsupportedOperation(self.add_conflicts, self)