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
class TestSmartServerBranchRequestGetConfigFile(tests.TestCaseWithMemoryTransport):

    def test_default(self):
        """With no file, we get empty content."""
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchGetConfigFile(backing)
        self.make_branch('.')
        content = b''
        self.assertEqual(smart_req.SmartServerResponse((b'ok',), content), request.execute(b''))

    def test_with_content(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchGetConfigFile(backing)
        branch = self.make_branch('.')
        branch._transport.put_bytes('branch.conf', b'foo bar baz')
        self.assertEqual(smart_req.SmartServerResponse((b'ok',), b'foo bar baz'), request.execute(b''))