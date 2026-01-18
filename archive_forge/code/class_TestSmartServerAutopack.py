from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
class TestSmartServerAutopack(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.smart_server = test_server.SmartTCPServer_for_testing()
        self.start_server(self.smart_server, self.get_server())
        client._SmartClient.hooks.install_named_hook('call', self.capture_hpss_call, None)
        self.hpss_calls = []

    def capture_hpss_call(self, params):
        self.hpss_calls.append(params.method)

    def get_format(self):
        return controldir.format_registry.make_controldir(self.format_name)

    def test_autopack_or_streaming_rpc_is_used_when_using_hpss(self):
        format = self.get_format()
        tree = self.make_branch_and_tree('local', format=format)
        self.make_branch_and_tree('remote', format=format)
        remote_branch_url = self.smart_server.get_url() + 'remote'
        remote_branch = controldir.ControlDir.open(remote_branch_url).open_branch()
        for x in range(9):
            tree.commit('commit %s' % x)
            tree.branch.push(remote_branch)
        self.hpss_calls = []
        tree.commit('commit triggering pack')
        tree.branch.push(remote_branch)
        autopack_calls = len([call for call in self.hpss_calls if call == b'PackRepository.autopack'])
        streaming_calls = len([call for call in self.hpss_calls if call in (b'Repository.insert_stream', b'Repository.insert_stream_1.19')])
        if autopack_calls:
            self.assertEqual(1, autopack_calls)
            self.assertEqual(0, streaming_calls)
        else:
            self.assertEqual(0, autopack_calls)
            self.assertEqual(2, streaming_calls)