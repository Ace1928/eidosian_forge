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
class TestRepositoryIterFilesBytes(TestRemoteRepository):
    """Test Repository.iter_file_bytes."""

    def test_single(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.iter_files_bytes', (b'quack/',), b'success', (b'ok',), iter([b'ok\x000', b'\n', zlib.compress(b'mydata' * 10)]))
        for identifier, byte_stream in repo.iter_files_bytes([(b'somefile', b'somerev', b'myid')]):
            self.assertEqual(b'myid', identifier)
            self.assertEqual(b''.join(byte_stream), b'mydata' * 10)

    def test_missing(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.iter_files_bytes', (b'quack/',), b'error', (b'RevisionNotPresent', b'somefile', b'somerev'), iter([b'absent\x00somefile\x00somerev\n']))
        self.assertRaises(errors.RevisionNotPresent, list, repo.iter_files_bytes([(b'somefile', b'somerev', b'myid')]))