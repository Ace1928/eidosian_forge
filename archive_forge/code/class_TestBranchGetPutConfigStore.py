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
class TestBranchGetPutConfigStore(RemoteBranchTestCase):

    def test_get_branch_conf(self):
        client = FakeClient()
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
        client.add_success_response_with_body(b'# config file body', b'ok')
        transport = MemoryTransport()
        branch = self.make_remote_branch(transport, client)
        config = branch.get_config_stack()
        config.get('email')
        config.get('log_format')
        self.assertEqual([('call', b'Branch.get_stacked_on_url', (b'memory:///',)), ('call_expecting_body', b'Branch.get_config_file', (b'memory:///',))], client._calls)

    def test_set_branch_conf(self):
        client = FakeClient()
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'memory:///',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.lock_write', (b'memory:///', b'', b''), b'success', (b'ok', b'branch token', b'repo token'))
        client.add_expected_call(b'Branch.get_config_file', (b'memory:///',), b'success', (b'ok',), b'# line 1\n')
        client.add_expected_call(b'Branch.get_config_file', (b'memory:///',), b'success', (b'ok',), b'# line 1\n')
        client.add_expected_call(b'Branch.put_config_file', (b'memory:///', b'branch token', b'repo token'), b'success', (b'ok',))
        client.add_expected_call(b'Branch.unlock', (b'memory:///', b'branch token', b'repo token'), b'success', (b'ok',))
        transport = MemoryTransport()
        branch = self.make_remote_branch(transport, client)
        branch.lock_write()
        config = branch.get_config_stack()
        config.set('email', 'The Dude <lebowski@example.com>')
        branch.unlock()
        self.assertFinished(client)
        self.assertEqual([('call', b'Branch.get_stacked_on_url', (b'memory:///',)), ('call', b'Branch.lock_write', (b'memory:///', b'', b'')), ('call_expecting_body', b'Branch.get_config_file', (b'memory:///',)), ('call_expecting_body', b'Branch.get_config_file', (b'memory:///',)), ('call_with_body_bytes_expecting_body', b'Branch.put_config_file', (b'memory:///', b'branch token', b'repo token'), b'# line 1\nemail = The Dude <lebowski@example.com>\n'), ('call', b'Branch.unlock', (b'memory:///', b'branch token', b'repo token'))], client._calls)