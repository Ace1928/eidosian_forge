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
class TestSmartServerRepositoryIterRevisions(tests.TestCaseWithMemoryTransport):

    def test_basic(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryIterRevisions(backing)
        tree = self.make_branch_and_memory_tree('.', format='2a')
        tree.lock_write()
        tree.add('')
        tree.commit('1st commit', rev_id=b'rev1')
        tree.commit('2nd commit', rev_id=b'rev2')
        tree.unlock()
        self.assertIs(None, request.execute(b''))
        response = request.do_body(b'rev1\nrev2')
        self.assertTrue(response.is_successful())
        self.assertEqual(response.args, (b'ok', b'10'))
        self.addCleanup(tree.branch.lock_read().unlock)
        entries = [zlib.compress(record.get_bytes_as('fulltext')) for record in tree.branch.repository.revisions.get_record_stream([(b'rev1',), (b'rev2',)], 'unordered', True)]
        contents = b''.join(response.body_stream)
        self.assertTrue(contents in (b''.join([entries[0], entries[1]]), b''.join([entries[1], entries[0]])))

    def test_missing(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryIterRevisions(backing)
        self.make_branch_and_memory_tree('.', format='2a')
        self.assertIs(None, request.execute(b''))
        response = request.do_body(b'rev1\nrev2')
        self.assertTrue(response.is_successful())
        self.assertEqual(response.args, (b'ok', b'10'))
        contents = b''.join(response.body_stream)
        self.assertEqual(contents, b'')