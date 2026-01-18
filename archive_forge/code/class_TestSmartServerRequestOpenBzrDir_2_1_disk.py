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
class TestSmartServerRequestOpenBzrDir_2_1_disk(TestCaseWithChrootedTransport):

    def test_present_with_workingtree(self):
        self.vfs_transport_factory = test_server.LocalURLServer
        backing = self.get_transport()
        request = smart_dir.SmartServerRequestOpenBzrDir_2_1(backing)
        bd = self.make_controldir('.')
        bd.create_repository()
        bd.create_branch()
        bd.create_workingtree()
        self.assertEqual(smart_req.SmartServerResponse((b'yes', b'yes')), request.execute(b''))