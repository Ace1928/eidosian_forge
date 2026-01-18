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
class TestRepositoryIsShared(TestRemoteRepository):

    def test_is_shared(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response(b'yes')
        result = repo.is_shared()
        self.assertEqual([('call', b'Repository.is_shared', (b'quack/',))], client._calls)
        self.assertEqual(True, result)

    def test_is_not_shared(self):
        transport_path = 'qwack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response(b'no')
        result = repo.is_shared()
        self.assertEqual([('call', b'Repository.is_shared', (b'qwack/',))], client._calls)
        self.assertEqual(False, result)