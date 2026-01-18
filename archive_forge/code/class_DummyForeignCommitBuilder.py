from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignCommitBuilder(PackCommitBuilder):

    def _generate_revision_if_needed(self, revid):
        mapping = DummyForeignVcsMapping(DummyForeignVcs())
        if self._lossy:
            self._new_revision_id = mapping.revision_id_foreign_to_bzr((b'%d' % self._timestamp, str(self._timezone).encode('ascii'), b'UNKNOWN'))
            self.random_revid = False
        elif revid is not None:
            self._new_revision_id = revid
            self.random_revid = False
        else:
            self._new_revision_id = self._gen_revision_id()
            self.random_revid = True