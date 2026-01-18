from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignProber(controldir.Prober):

    @classmethod
    def probe_transport(klass, transport):
        """Return the .bzrdir style format present in a directory."""
        if not transport.has('.dummy'):
            raise errors.NotBranchError(path=transport.base)
        return DummyForeignVcsDirFormat()

    @classmethod
    def known_formats(cls):
        return [DummyForeignVcsDirFormat()]