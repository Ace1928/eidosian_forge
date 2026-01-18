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
class TestSmartServerRepositoryAddSignatureText(tests.TestCaseWithMemoryTransport):

    def test_add_text(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryAddSignatureText(backing)
        tree = self.make_branch_and_memory_tree('.')
        write_token = tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add('')
        tree.commit('Message', rev_id=b'rev1')
        tree.branch.repository.start_write_group()
        write_group_tokens = tree.branch.repository.suspend_write_group()
        self.assertEqual(None, request.execute(b'', write_token, b'rev1', *[token.encode('utf-8') for token in write_group_tokens]))
        response = request.do_body(b'somesignature')
        self.assertTrue(response.is_successful())
        self.assertEqual(response.args[0], b'ok')
        write_group_tokens = [token.decode('utf-8') for token in response.args[1:]]
        tree.branch.repository.resume_write_group(write_group_tokens)
        tree.branch.repository.commit_write_group()
        tree.unlock()
        self.assertEqual(b'somesignature', tree.branch.repository.get_signature_text(b'rev1'))