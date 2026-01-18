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
class TestRepositoryHasSignatureForRevisionId(TestRemoteRepository):

    def test_has_signature_for_revision_id(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response(b'yes')
        result = repo.has_signature_for_revision_id(b'A')
        self.assertEqual([('call', b'Repository.has_signature_for_revision_id', (b'quack/', b'A'))], client._calls)
        self.assertEqual(True, result)

    def test_is_not_shared(self):
        transport_path = 'qwack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response(b'no')
        result = repo.has_signature_for_revision_id(b'A')
        self.assertEqual([('call', b'Repository.has_signature_for_revision_id', (b'qwack/', b'A'))], client._calls)
        self.assertEqual(False, result)