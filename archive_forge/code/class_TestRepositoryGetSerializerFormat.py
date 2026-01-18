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
class TestRepositoryGetSerializerFormat(TestRemoteRepository):

    def test_get_serializer_format(self):
        transport_path = 'hill'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_success_response(b'ok', b'7')
        self.assertEqual(b'7', repo.get_serializer_format())
        self.assertEqual([('call', b'VersionedFileRepository.get_serializer_format', (b'hill/',))], client._calls)