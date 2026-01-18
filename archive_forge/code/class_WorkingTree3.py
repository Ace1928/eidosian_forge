import errno
from .. import errors, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import MutableTree
from ..transport.local import LocalTransport
from . import bzrdir, hashcache, inventory
from . import transform as bzr_transform
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
class WorkingTree3(PreDirStateWorkingTree):
    """This is the Format 3 working tree.

    This differs from the base WorkingTree by:
     - having its own file lock
     - having its own last-revision property.

    This is new in bzr 0.8
    """

    def _last_revision(self):
        """See Mutable.last_revision."""
        with self.lock_read():
            try:
                return self._transport.get_bytes('last-revision')
            except _mod_transport.NoSuchFile:
                return _mod_revision.NULL_REVISION

    def _change_last_revision(self, revision_id):
        """See WorkingTree._change_last_revision."""
        if revision_id is None or revision_id == _mod_revision.NULL_REVISION:
            try:
                self._transport.delete('last-revision')
            except _mod_transport.NoSuchFile:
                pass
            return False
        else:
            self._transport.put_bytes('last-revision', revision_id, mode=self.controldir._get_file_mode())
            return True

    def _get_check_refs(self):
        """Return the references needed to perform a check of this tree."""
        return [('trees', self.last_revision())]

    def unlock(self):
        if self._control_files._lock_count == 1:
            self._cleanup()
            if self._inventory_is_modified:
                self.flush()
            self._write_hashcache_if_dirty()
        try:
            return self._control_files.unlock()
        finally:
            self.branch.unlock()