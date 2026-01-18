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
class PreDirStateWorkingTree(InventoryWorkingTree):

    def __init__(self, basedir='.', *args, **kwargs):
        super().__init__(basedir, *args, **kwargs)
        wt_trans = self.controldir.get_workingtree_transport(None)
        cache_filename = wt_trans.local_abspath('stat-cache')
        self._hashcache = hashcache.HashCache(basedir, cache_filename, self.controldir._get_file_mode(), self._content_filter_stack_provider())
        hc = self._hashcache
        hc.read()
        if hc.needs_write:
            trace.mutter('write hc')
            hc.write()

    def _write_hashcache_if_dirty(self):
        """Write out the hashcache if it is dirty."""
        if self._hashcache.needs_write:
            try:
                self._hashcache.write()
            except OSError as e:
                if e.errno not in (errno.EPERM, errno.EACCES):
                    raise
                trace.mutter('Could not write hashcache for %s\nError: %s', self._hashcache.cache_file_name(), osutils.safe_unicode(e.args[1]))

    def get_file_sha1(self, path, stat_value=None):
        with self.lock_read():
            if not self.is_versioned(path):
                raise _mod_transport.NoSuchFile(path)
            return self._hashcache.get_sha1(path, stat_value)