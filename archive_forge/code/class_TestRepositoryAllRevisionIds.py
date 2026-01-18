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
class TestRepositoryAllRevisionIds(TestRemoteRepository):

    def test_empty(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(b'', b'ok')
        self.assertEqual([], repo.all_revision_ids())
        self.assertEqual([('call_expecting_body', b'Repository.all_revision_ids', (b'quack/',))], client._calls)

    def test_with_some_content(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(b'rev1\nrev2\nanotherrev\n', b'ok')
        self.assertEqual({b'rev1', b'rev2', b'anotherrev'}, set(repo.all_revision_ids()))
        self.assertEqual([('call_expecting_body', b'Repository.all_revision_ids', (b'quack/',))], client._calls)