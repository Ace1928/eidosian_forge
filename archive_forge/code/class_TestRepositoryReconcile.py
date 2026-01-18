import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class TestRepositoryReconcile(TestRemoteRepository):

    def test_reconcile(self):
        transport_path = 'hill'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        body = b'garbage_inventories: 2\ninconsistent_parents: 3\n'
        client.add_expected_call(b'Repository.lock_write', (b'hill/', b''), b'success', (b'ok', b'a token'))
        client.add_success_response_with_body(body, b'ok')
        reconciler = repo.reconcile()
        self.assertEqual([('call', b'Repository.lock_write', (b'hill/', b'')), ('call_expecting_body', b'Repository.reconcile', (b'hill/', b'a token'))], client._calls)
        self.assertEqual(2, reconciler.garbage_inventories)
        self.assertEqual(3, reconciler.inconsistent_parents)