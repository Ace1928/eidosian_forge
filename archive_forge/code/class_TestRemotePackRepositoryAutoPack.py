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
class TestRemotePackRepositoryAutoPack(TestRemoteRepository):
    """Tests for RemoteRepository.autopack implementation."""

    def test_ok(self):
        """When the server returns 'ok' and there's no _real_repository, then
        nothing else happens: the autopack method is done.
        """
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'PackRepository.autopack', (b'quack/',), b'success', (b'ok',))
        repo.autopack()
        self.assertFinished(client)

    def test_ok_with_real_repo(self):
        """When the server returns 'ok' and there is a _real_repository, then
        the _real_repository's reload_pack_name's method will be called.
        """
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'PackRepository.autopack', (b'quack/',), b'success', (b'ok',))
        repo._real_repository = _StubRealPackRepository(client._calls)
        repo.autopack()
        self.assertEqual([('call', b'PackRepository.autopack', (b'quack/',)), ('pack collection reload_pack_names',)], client._calls)

    def test_backwards_compatibility(self):
        """If the server does not recognise the PackRepository.autopack verb,
        fallback to the real_repository's implementation.
        """
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_unknown_method_response(b'PackRepository.autopack')

        def stub_ensure_real():
            client._calls.append(('_ensure_real',))
            repo._real_repository = _StubRealPackRepository(client._calls)
        repo._ensure_real = stub_ensure_real
        repo.autopack()
        self.assertEqual([('call', b'PackRepository.autopack', (b'quack/',)), ('_ensure_real',), ('pack collection autopack',)], client._calls)

    def test_oom_error_reporting(self):
        """An out-of-memory condition on the server is reported clearly"""
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'PackRepository.autopack', (b'quack/',), b'error', (b'MemoryError',))
        err = self.assertRaises(errors.BzrError, repo.autopack)
        self.assertContainsRe(str(err), '^remote server out of mem')