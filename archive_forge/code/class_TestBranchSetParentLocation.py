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
class TestBranchSetParentLocation(RemoteBranchTestCase):

    def test_no_parent(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.set_parent_location', (b'quack/', b'b', b'r', b''), b'success', ())
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        branch._lock_token = b'b'
        branch._repo_lock_token = b'r'
        branch._set_parent_location(None)
        self.assertFinished(client)

    def test_parent(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'kwaak/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.set_parent_location', (b'kwaak/', b'b', b'r', b'foo'), b'success', ())
        transport.mkdir('kwaak')
        transport = transport.clone('kwaak')
        branch = self.make_remote_branch(transport, client)
        branch._lock_token = b'b'
        branch._repo_lock_token = b'r'
        branch._set_parent_location('foo')
        self.assertFinished(client)

    def test_backwards_compat(self):
        self.setup_smart_server_with_call_log()
        branch = self.make_branch('.')
        self.reset_smart_call_log()
        verb = b'Branch.set_parent_location'
        self.disable_verb(verb)
        branch.set_parent('http://foo/')
        self.assertLength(14, self.hpss_calls)