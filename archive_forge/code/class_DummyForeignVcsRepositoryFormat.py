from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsRepositoryFormat(groupcompress_repo.RepositoryFormat2a):
    repository_class = DummyForeignVcsRepository
    _commit_builder_class = DummyForeignCommitBuilder

    @classmethod
    def get_format_string(cls):
        return b'Dummy Foreign Vcs Repository'

    def get_format_description(self):
        return 'Dummy Foreign Vcs Repository'