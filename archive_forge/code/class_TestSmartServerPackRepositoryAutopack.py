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
class TestSmartServerPackRepositoryAutopack(tests.TestCaseWithTransport):

    def make_repo_needing_autopacking(self, path='.'):
        tree = self.make_branch_and_tree('.', format='pack-0.92')
        repo = tree.branch.repository
        repo._pack_collection._max_pack_count = lambda count: count
        for x in range(10):
            tree.commit('commit %s' % x)
        self.assertEqual(10, len(repo._pack_collection.names()))
        del repo._pack_collection._max_pack_count
        return repo

    def test_autopack_needed(self):
        repo = self.make_repo_needing_autopacking()
        repo.lock_write()
        self.addCleanup(repo.unlock)
        backing = self.get_transport()
        request = smart_packrepo.SmartServerPackRepositoryAutopack(backing)
        response = request.execute(b'')
        self.assertEqual(smart_req.SmartServerResponse((b'ok',)), response)
        repo._pack_collection.reload_pack_names()
        self.assertEqual(1, len(repo._pack_collection.names()))

    def test_autopack_not_needed(self):
        tree = self.make_branch_and_tree('.', format='pack-0.92')
        repo = tree.branch.repository
        repo.lock_write()
        self.addCleanup(repo.unlock)
        for x in range(9):
            tree.commit('commit %s' % x)
        backing = self.get_transport()
        request = smart_packrepo.SmartServerPackRepositoryAutopack(backing)
        response = request.execute(b'')
        self.assertEqual(smart_req.SmartServerResponse((b'ok',)), response)
        repo._pack_collection.reload_pack_names()
        self.assertEqual(9, len(repo._pack_collection.names()))

    def test_autopack_on_nonpack_format(self):
        """A request to autopack a non-pack repo is a no-op."""
        repo = self.make_repository('.', format='knit')
        backing = self.get_transport()
        request = smart_packrepo.SmartServerPackRepositoryAutopack(backing)
        response = request.execute(b'')
        self.assertEqual(smart_req.SmartServerResponse((b'ok',)), response)