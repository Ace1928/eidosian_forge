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
class TestRepositoryGetRevisionGraph(TestRemoteRepository):

    def test_null_revision(self):
        transport_path = 'empty'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response(b'notused')
        result = repo._get_revision_graph(NULL_REVISION)
        self.assertEqual([], client._calls)
        self.assertEqual({}, result)

    def test_none_revision(self):
        r1 = 'ำ'.encode()
        r2 = 'ණ'.encode()
        lines = [b' '.join([r2, r1]), r1]
        encoded_body = b'\n'.join(lines)
        transport_path = 'sinhala'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(encoded_body, b'ok')
        result = repo._get_revision_graph(None)
        self.assertEqual([('call_expecting_body', b'Repository.get_revision_graph', (b'sinhala/', b''))], client._calls)
        self.assertEqual({r1: (), r2: (r1,)}, result)

    def test_specific_revision(self):
        r11 = 'ำ'.encode()
        r12 = 'É'.encode()
        r2 = 'ණ'.encode()
        lines = [b' '.join([r2, r11, r12]), r11, r12]
        encoded_body = b'\n'.join(lines)
        transport_path = 'sinhala'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response_with_body(encoded_body, b'ok')
        result = repo._get_revision_graph(r2)
        self.assertEqual([('call_expecting_body', b'Repository.get_revision_graph', (b'sinhala/', r2))], client._calls)
        self.assertEqual({r11: (), r12: (), r2: (r11, r12)}, result)

    def test_no_such_revision(self):
        revid = b'123'
        transport_path = 'sinhala'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_error_response(b'nosuchrevision', revid)
        self.assertRaises(errors.NoSuchRevision, repo._get_revision_graph, revid)
        self.assertEqual([('call_expecting_body', b'Repository.get_revision_graph', (b'sinhala/', revid))], client._calls)

    def test_unexpected_error(self):
        revid = '123'
        transport_path = 'sinhala'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_error_response(b'AnUnexpectedError')
        e = self.assertRaises(UnknownErrorFromSmartServer, repo._get_revision_graph, revid)
        self.assertEqual((b'AnUnexpectedError',), e.error_tuple)