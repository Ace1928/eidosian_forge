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
class TestSmartServerRepositoryGetInventories(tests.TestCaseWithTransport):

    def _get_serialized_inventory_delta(self, repository, base_revid, revid):
        base_inv = repository.revision_tree(base_revid).root_inventory
        inv = repository.revision_tree(revid).root_inventory
        inv_delta = inv._make_delta(base_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(True, True)
        return b''.join(serializer.delta_to_lines(base_revid, revid, inv_delta))

    def test_single(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetInventories(backing)
        t = self.make_branch_and_tree('.', format='2a')
        self.addCleanup(t.lock_write().unlock)
        self.build_tree_contents([('file', b'somecontents')])
        t.add(['file'], ids=[b'thefileid'])
        t.commit(rev_id=b'somerev', message='add file')
        self.assertIs(None, request.execute(b'', b'unordered'))
        response = request.do_body(b'somerev\n')
        self.assertTrue(response.is_successful())
        self.assertEqual(response.args, (b'ok',))
        stream = [('inventory-deltas', [versionedfile.FulltextContentFactory(b'somerev', None, None, self._get_serialized_inventory_delta(t.branch.repository, b'null:', b'somerev'))])]
        fmt = controldir.format_registry.get('2a')().repository_format
        self.assertEqual(b''.join(response.body_stream), b''.join(smart_repo._stream_to_byte_stream(stream, fmt)))

    def test_empty(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryGetInventories(backing)
        t = self.make_branch_and_tree('.', format='2a')
        self.addCleanup(t.lock_write().unlock)
        self.build_tree_contents([('file', b'somecontents')])
        t.add(['file'], ids=[b'thefileid'])
        t.commit(rev_id=b'somerev', message='add file')
        self.assertIs(None, request.execute(b'', b'unordered'))
        response = request.do_body(b'')
        self.assertTrue(response.is_successful())
        self.assertEqual(response.args, (b'ok',))
        self.assertEqual(b''.join(response.body_stream), b'Bazaar pack format 1 (introduced in 0.18)\nB54\n\nBazaar repository format 2a (needs bzr 1.16 or later)\nE')