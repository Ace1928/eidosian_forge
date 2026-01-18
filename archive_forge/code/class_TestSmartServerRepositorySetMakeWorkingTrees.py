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
class TestSmartServerRepositorySetMakeWorkingTrees(tests.TestCaseWithMemoryTransport):

    def test_set_false(self):
        backing = self.get_transport()
        repo = self.make_repository('.', shared=True)
        repo.set_make_working_trees(True)
        request_class = smart_repo.SmartServerRepositorySetMakeWorkingTrees
        request = request_class(backing)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok',)), request.execute(b'', b'False'))
        repo = repo.controldir.open_repository()
        self.assertFalse(repo.make_working_trees())

    def test_set_true(self):
        backing = self.get_transport()
        repo = self.make_repository('.', shared=True)
        repo.set_make_working_trees(False)
        request_class = smart_repo.SmartServerRepositorySetMakeWorkingTrees
        request = request_class(backing)
        self.assertEqual(smart_req.SuccessfulSmartServerResponse((b'ok',)), request.execute(b'', b'True'))
        repo = repo.controldir.open_repository()
        self.assertTrue(repo.make_working_trees())