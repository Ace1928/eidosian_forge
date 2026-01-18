from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsBranchFormat(bzrbranch.BzrBranchFormat6):

    @classmethod
    def get_format_string(cls):
        return b'Branch for Testing'

    @property
    def _matchingcontroldir(self):
        return DummyForeignVcsDirFormat()

    def open(self, a_controldir, name=None, _found=False, ignore_fallbacks=False, found_repository=None):
        if name is None:
            name = a_controldir._get_selected_branch()
        if not _found:
            raise NotImplementedError
        try:
            transport = a_controldir.get_branch_transport(None, name=name)
            control_files = lockable_files.LockableFiles(transport, 'lock', lockdir.LockDir)
            if found_repository is None:
                found_repository = a_controldir.find_repository()
            return DummyForeignVcsBranch(_format=self, _control_files=control_files, a_controldir=a_controldir, _repository=found_repository, name=name)
        except _mod_transport.NoSuchFile:
            raise errors.NotBranchError(path=transport.base)