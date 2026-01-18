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
class TestSmartServerBranchRequestRevisionIdToRevno(tests.TestCaseWithMemoryTransport):

    def test_null(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestRevisionIdToRevno(backing)
        self.make_branch('.')
        self.assertEqual(smart_req.SmartServerResponse((b'ok', b'0')), request.execute(b'', b'null:'))

    def test_ghost_revision(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestRevisionIdToRevno(backing)
        branch = self.make_branch('.')

        def revision_id_to_dotted_revno(revid):
            raise errors.GhostRevisionsHaveNoRevno(revid, b'ghost-revid')
        self.overrideAttr(branch, 'revision_id_to_dotted_revno', revision_id_to_dotted_revno)
        self.assertEqual(smart_req.FailedSmartServerResponse((b'GhostRevisionsHaveNoRevno', b'revid', b'ghost-revid')), request.do_with_branch(branch, b'revid'))

    def test_simple(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestRevisionIdToRevno(backing)
        tree = self.make_branch_and_memory_tree('.')
        tree.lock_write()
        tree.add('')
        r1 = tree.commit('1st commit')
        tree.unlock()
        self.assertEqual(smart_req.SmartServerResponse((b'ok', b'1')), request.execute(b'', r1))

    def test_not_found(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestRevisionIdToRevno(backing)
        self.make_branch('.')
        self.assertEqual(smart_req.FailedSmartServerResponse((b'NoSuchRevision', b'idontexist')), request.execute(b'', b'idontexist'))