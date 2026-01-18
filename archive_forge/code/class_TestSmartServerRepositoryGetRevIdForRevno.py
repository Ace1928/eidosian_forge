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
class TestSmartServerRepositoryGetRevIdForRevno(tests.TestCaseWithMemoryTransport):

    def test_revno_found(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetRevIdForRevno(backing)
        tree = self.make_branch_and_memory_tree('.')
        tree.lock_write()
        tree.add('')
        rev1_id_utf8 = 'È'.encode()
        rev2_id_utf8 = 'É'.encode()
        tree.commit('1st commit', rev_id=rev1_id_utf8)
        tree.commit('2nd commit', rev_id=rev2_id_utf8)
        tree.unlock()
        self.assertEqual(smart_req.SmartServerResponse((b'ok', rev1_id_utf8)), request.execute(b'', 1, (2, rev2_id_utf8)))

    def test_known_revid_missing(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetRevIdForRevno(backing)
        self.make_repository('.')
        self.assertEqual(smart_req.FailedSmartServerResponse((b'nosuchrevision', b'ghost')), request.execute(b'', 1, (2, b'ghost')))

    def test_history_incomplete(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetRevIdForRevno(backing)
        parent = self.make_branch_and_memory_tree('parent', format='1.9')
        parent.lock_write()
        parent.add([''], ids=[b'TREE_ROOT'])
        parent.commit(message='first commit')
        r2 = parent.commit(message='second commit')
        parent.unlock()
        local = self.make_branch_and_memory_tree('local', format='1.9')
        local.branch.pull(parent.branch)
        local.set_parent_ids([r2])
        r3 = local.commit(message='local commit')
        local.branch.create_clone_on_transport(self.get_transport('stacked'), stacked_on=self.get_url('parent'))
        self.assertEqual(smart_req.SmartServerResponse((b'history-incomplete', 2, r2)), request.execute(b'stacked', 1, (3, r3)))