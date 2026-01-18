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
class TestSetLastRevisionVerbMixin:
    """Mixin test case for verbs that implement set_last_revision."""

    def test_set_null_to_null(self):
        """An empty branch can have its last revision set to b'null:'."""
        self.assertRequestSucceeds(b'null:', 0)

    def test_NoSuchRevision(self):
        """If the revision_id is not present, the verb returns NoSuchRevision.
        """
        revision_id = b'non-existent revision'
        self.assertEqual(smart_req.FailedSmartServerResponse((b'NoSuchRevision', revision_id)), self.set_last_revision(revision_id, 1))

    def make_tree_with_two_commits(self):
        self.tree.lock_write()
        self.tree.add('')
        rev_id_utf8 = 'È'.encode()
        self.tree.commit('1st commit', rev_id=rev_id_utf8)
        self.tree.commit('2nd commit', rev_id=b'rev-2')
        self.tree.unlock()

    def test_branch_last_revision_info_is_updated(self):
        """A branch's tip can be set to a revision that is present in its
        repository.
        """
        self.make_tree_with_two_commits()
        rev_id_utf8 = 'È'.encode()
        self.tree.branch.set_last_revision_info(0, b'null:')
        self.assertEqual((0, b'null:'), self.tree.branch.last_revision_info())
        self.assertRequestSucceeds(rev_id_utf8, 1)
        self.assertEqual((1, rev_id_utf8), self.tree.branch.last_revision_info())

    def test_branch_last_revision_info_rewind(self):
        """A branch's tip can be set to a revision that is an ancestor of the
        current tip.
        """
        self.make_tree_with_two_commits()
        rev_id_utf8 = 'È'.encode()
        self.assertEqual((2, b'rev-2'), self.tree.branch.last_revision_info())
        self.assertRequestSucceeds(rev_id_utf8, 1)
        self.assertEqual((1, rev_id_utf8), self.tree.branch.last_revision_info())

    def test_TipChangeRejected(self):
        """If a pre_change_branch_tip hook raises TipChangeRejected, the verb
        returns TipChangeRejected.
        """
        rejection_message = 'rejection message‽'

        def hook_that_rejects(params):
            raise errors.TipChangeRejected(rejection_message)
        _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', hook_that_rejects, None)
        self.assertEqual(smart_req.FailedSmartServerResponse((b'TipChangeRejected', rejection_message.encode('utf-8'))), self.set_last_revision(b'null:', 0))