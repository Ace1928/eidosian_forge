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
class TestBranchSetTagsBytes(RemoteBranchTestCase):

    def test_trivial(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.set_tags_bytes', (b'quack/', b'branch token', b'repo token'), b'success', ('',))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        self.lock_remote_branch(branch)
        branch._set_tags_bytes(b'tags bytes')
        self.assertFinished(client)
        self.assertEqual(b'tags bytes', client._calls[-1][-1])

    def test_backwards_compatible(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.set_tags_bytes', (b'quack/', b'branch token', b'repo token'), b'unknown', (b'Branch.set_tags_bytes',))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        self.lock_remote_branch(branch)

        class StubRealBranch:

            def __init__(self):
                self.calls = []

            def _set_tags_bytes(self, bytes):
                self.calls.append(('set_tags_bytes', bytes))
        real_branch = StubRealBranch()
        branch._real_branch = real_branch
        branch._set_tags_bytes(b'tags bytes')
        branch._set_tags_bytes(b'tags bytes')
        self.assertFinished(client)
        self.assertEqual([('set_tags_bytes', b'tags bytes')] * 2, real_branch.calls)