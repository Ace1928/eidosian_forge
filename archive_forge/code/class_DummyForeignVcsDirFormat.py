from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsDirFormat(bzrdir.BzrDirMetaFormat1):
    """BzrDirFormat for the dummy foreign VCS."""

    @classmethod
    def get_format_string(cls):
        return b'A Dummy VCS Dir'

    @classmethod
    def get_format_description(cls):
        return 'A Dummy VCS Dir'

    @classmethod
    def is_supported(cls):
        return True

    def get_branch_format(self):
        return DummyForeignVcsBranchFormat()

    @property
    def repository_format(self):
        return DummyForeignVcsRepositoryFormat()

    def initialize_on_transport(self, transport):
        """Initialize a new bzrdir in the base directory of a Transport."""
        temp_control = lockable_files.LockableFiles(transport, '', lockable_files.TransportLock)
        temp_control._transport.mkdir('.dummy', mode=temp_control._dir_mode)
        del temp_control
        bzrdir_transport = transport.clone('.dummy')
        control_files = lockable_files.LockableFiles(bzrdir_transport, self._lock_file_name, self._lock_class)
        control_files.create_lock()
        return self.open(transport, _found=True)

    def _open(self, transport):
        return DummyForeignVcsDir(transport, self)