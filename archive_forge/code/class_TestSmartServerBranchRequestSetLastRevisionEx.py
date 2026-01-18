import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
class TestSmartServerBranchRequestSetLastRevisionEx(SetLastRevisionTestBase, TestSetLastRevisionVerbMixin):
    """Tests for Branch.set_last_revision_ex verb."""
    request_class = smart_branch.SmartServerBranchRequestSetLastRevisionEx

    def _set_last_revision(self, revision_id, revno, branch_token, repo_token):
        return self.request.execute(b'', branch_token, repo_token, revision_id, 0, 0)

    def assertRequestSucceeds(self, revision_id, revno):
        response = self.set_last_revision(revision_id, revno)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok', revno, revision_id)), response)

    def test_branch_last_revision_info_rewind(self):
        """A branch's tip can be set to a revision that is an ancestor of the
        current tip, but only if allow_overwrite_descendant is passed.
        """
        self.make_tree_with_two_commits()
        rev_id_utf8 = 'È'.encode()
        self.assertEqual((2, b'rev-2'), self.tree.branch.last_revision_info())
        branch_token, repo_token = self.lock_branch()
        response = self.request.execute(b'', branch_token, repo_token, rev_id_utf8, 0, 0)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok', 2, b'rev-2')), response)
        self.assertEqual((2, b'rev-2'), self.tree.branch.last_revision_info())
        response = self.request.execute(b'', branch_token, repo_token, rev_id_utf8, 0, 1)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok', 1, rev_id_utf8)), response)
        self.unlock_branch()
        self.assertEqual((1, rev_id_utf8), self.tree.branch.last_revision_info())

    def make_branch_with_divergent_history(self):
        """Make a branch with divergent history in its repo.

        The branch's tip will be 'child-2', and the repo will also contain
        'child-1', which diverges from a common base revision.
        """
        self.tree.lock_write()
        self.tree.add('')
        self.tree.commit('1st commit')
        revno_1, revid_1 = self.tree.branch.last_revision_info()
        self.tree.commit('2nd commit', rev_id=b'child-1')
        self.tree.branch.set_last_revision_info(revno_1, revid_1)
        self.tree.set_parent_ids([revid_1])
        self.tree.commit('2nd commit', rev_id=b'child-2')
        self.tree.unlock()

    def test_not_allow_diverged(self):
        """If allow_diverged is not passed, then setting a divergent history
        returns a Diverged error.
        """
        self.make_branch_with_divergent_history()
        self.assertEqual(smart_req.FailedSmartServerResponse((b'Diverged',)), self.set_last_revision(b'child-1', 2))
        self.assertEqual(b'child-2', self.tree.branch.last_revision())

    def test_allow_diverged(self):
        """If allow_diverged is passed, then setting a divergent history
        succeeds.
        """
        self.make_branch_with_divergent_history()
        branch_token, repo_token = self.lock_branch()
        response = self.request.execute(b'', branch_token, repo_token, b'child-1', 1, 0)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok', 2, b'child-1')), response)
        self.unlock_branch()
        self.assertEqual(b'child-1', self.tree.branch.last_revision())