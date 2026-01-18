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
class TestBzrDirOpenBranch(TestRemote):

    def test_backwards_compat(self):
        self.setup_smart_server_with_call_log()
        self.make_branch('.')
        a_dir = BzrDir.open(self.get_url('.'))
        self.reset_smart_call_log()
        verb = b'BzrDir.open_branchV3'
        self.disable_verb(verb)
        a_dir.open_branch()
        call_count = len([call for call in self.hpss_calls if call.call.method == verb])
        self.assertEqual(1, call_count)

    def test_branch_present(self):
        reference_format = self.get_repo_format()
        network_name = reference_format.network_name()
        branch_network_name = self.get_branch_format().network_name()
        transport = MemoryTransport()
        transport.mkdir('quack')
        transport = transport.clone('quack')
        client = FakeClient(transport.base)
        client.add_expected_call(b'BzrDir.open_branchV3', (b'quack/',), b'success', (b'branch', branch_network_name))
        client.add_expected_call(b'BzrDir.find_repositoryV3', (b'quack/',), b'success', (b'ok', b'', b'no', b'no', b'no', network_name))
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        bzrdir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        result = bzrdir.open_branch()
        self.assertIsInstance(result, RemoteBranch)
        self.assertEqual(bzrdir, result.controldir)
        self.assertFinished(client)

    def test_branch_missing(self):
        transport = MemoryTransport()
        transport.mkdir('quack')
        transport = transport.clone('quack')
        client = FakeClient(transport.base)
        client.add_error_response(b'nobranch')
        bzrdir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        self.assertRaises(errors.NotBranchError, bzrdir.open_branch)
        self.assertEqual([('call', b'BzrDir.open_branchV3', (b'quack/',))], client._calls)

    def test__get_tree_branch(self):
        calls = []

        def open_branch(name=None, possible_transports=None):
            calls.append('Called')
            return 'a-branch'
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        bzrdir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        bzrdir.open_branch = open_branch
        self.assertEqual((None, 'a-branch'), bzrdir._get_tree_branch())
        self.assertEqual(['Called'], calls)
        self.assertEqual([], client._calls)

    def test_url_quoting_of_path(self):
        transport = RemoteTCPTransport('bzr://localhost/~hello/')
        client = FakeClient(transport.base)
        reference_format = self.get_repo_format()
        network_name = reference_format.network_name()
        branch_network_name = self.get_branch_format().network_name()
        client.add_expected_call(b'BzrDir.open_branchV3', (b'~hello/',), b'success', (b'branch', branch_network_name))
        client.add_expected_call(b'BzrDir.find_repositoryV3', (b'~hello/',), b'success', (b'ok', b'', b'no', b'no', b'no', network_name))
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'~hello/',), b'error', (b'NotStacked',))
        bzrdir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        bzrdir.open_branch()
        self.assertFinished(client)

    def check_open_repository(self, rich_root, subtrees, external_lookup=b'no'):
        reference_format = self.get_repo_format()
        network_name = reference_format.network_name()
        transport = MemoryTransport()
        transport.mkdir('quack')
        transport = transport.clone('quack')
        if rich_root:
            rich_response = b'yes'
        else:
            rich_response = b'no'
        if subtrees:
            subtree_response = b'yes'
        else:
            subtree_response = b'no'
        client = FakeClient(transport.base)
        client.add_success_response(b'ok', b'', rich_response, subtree_response, external_lookup, network_name)
        bzrdir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        result = bzrdir.open_repository()
        self.assertEqual([('call', b'BzrDir.find_repositoryV3', (b'quack/',))], client._calls)
        self.assertIsInstance(result, RemoteRepository)
        self.assertEqual(bzrdir, result.controldir)
        self.assertEqual(rich_root, result._format.rich_root_data)
        self.assertEqual(subtrees, result._format.supports_tree_reference)

    def test_open_repository_sets_format_attributes(self):
        self.check_open_repository(True, True)
        self.check_open_repository(False, True)
        self.check_open_repository(True, False)
        self.check_open_repository(False, False)
        self.check_open_repository(False, False, b'yes')

    def test_old_server(self):
        """RemoteBzrDirFormat should fail to probe if the server version is too
        old.
        """
        self.assertRaises(errors.NotBranchError, RemoteBzrProber.probe_transport, OldServerTransport())