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
class TestSmartServerRepositoryGetRevisionGraph(tests.TestCaseWithMemoryTransport):

    def test_none_argument(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetRevisionGraph(backing)
        tree = self.make_branch_and_memory_tree('.')
        tree.lock_write()
        tree.add('')
        r1 = tree.commit('1st commit')
        r2 = tree.commit('2nd commit', rev_id='È'.encode())
        tree.unlock()
        lines = sorted([b' '.join([r2, r1]), r1])
        response = request.execute(b'', b'')
        response.body = b'\n'.join(sorted(response.body.split(b'\n')))
        self.assertEqual(smart_req.SmartServerResponse((b'ok',), b'\n'.join(lines)), response)

    def test_specific_revision_argument(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetRevisionGraph(backing)
        tree = self.make_branch_and_memory_tree('.')
        tree.lock_write()
        tree.add('')
        rev_id_utf8 = 'É'.encode()
        tree.commit('1st commit', rev_id=rev_id_utf8)
        tree.commit('2nd commit', rev_id='È'.encode())
        tree.unlock()
        self.assertEqual(smart_req.SmartServerResponse((b'ok',), rev_id_utf8), request.execute(b'', rev_id_utf8))

    def test_no_such_revision(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetRevisionGraph(backing)
        tree = self.make_branch_and_memory_tree('.')
        tree.lock_write()
        tree.add('')
        tree.commit('1st commit')
        tree.unlock()
        self.assertEqual(smart_req.SmartServerResponse((b'nosuchrevision', b'missingrevision'), b''), request.execute(b'', b'missingrevision'))