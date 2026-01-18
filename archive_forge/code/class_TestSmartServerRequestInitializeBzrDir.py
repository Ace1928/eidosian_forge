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
class TestSmartServerRequestInitializeBzrDir(tests.TestCaseWithMemoryTransport):

    def test_empty_dir(self):
        """Initializing an empty dir should succeed and do it."""
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestInitializeBzrDir(backing)
        self.assertEqual(smart_req.SmartServerResponse((b'ok',)), request.execute(b''))
        made_dir = controldir.ControlDir.open_from_transport(backing)
        self.assertRaises(errors.NoWorkingTree, made_dir.open_workingtree)
        self.assertRaises(errors.NotBranchError, made_dir.open_branch)
        self.assertRaises(errors.NoRepositoryPresent, made_dir.open_repository)

    def test_missing_dir(self):
        """Initializing a missing directory should fail like the bzrdir api."""
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestInitializeBzrDir(backing)
        self.assertRaises(transport.NoSuchFile, request.execute, b'subdir')

    def test_initialized_dir(self):
        """Initializing an extant bzrdir should fail like the bzrdir api."""
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestInitializeBzrDir(backing)
        self.make_controldir('subdir')
        self.assertRaises(errors.AlreadyControlDirError, request.execute, b'subdir')