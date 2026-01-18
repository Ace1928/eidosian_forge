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
class TestSmartServerRepositoryRequest(tests.TestCaseWithMemoryTransport):

    def test_no_repository(self):
        """Raise NoRepositoryPresent when there is a bzrdir and no repo."""
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryRequest(backing)
        self.make_repository('.', shared=True)
        self.make_controldir('subdir')
        self.assertRaises(errors.NoRepositoryPresent, request.execute, b'subdir')