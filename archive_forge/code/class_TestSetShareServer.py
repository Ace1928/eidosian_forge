from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestSetShareServer(TestShareServer):

    def setUp(self):
        super(TestSetShareServer, self).setUp()
        self.share_server = manila_fakes.FakeShareServer.create_one_server(methods={'reset_task_state': None})
        self.servers_mock.get.return_value = self.share_server
        self.cmd = osc_share_servers.SetShareServer(self.app, None)

    def test_share_server_set_status(self):
        arglist = [self.share_server.id, '--status', 'active']
        verifylist = [('share_server', self.share_server.id), ('status', 'active')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.reset_state.assert_called_with(self.share_server, parsed_args.status)
        self.assertIsNone(result)

    def test_share_server_set_task_state(self):
        arglist = [self.share_server.id, '--task-state', 'migration_in_progress']
        verifylist = [('share_server', self.share_server.id), ('task_state', 'migration_in_progress')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.reset_task_state.assert_called_with(self.share_server, parsed_args.task_state)
        self.assertIsNone(result)

    def test_share_server_set_status_exception(self):
        arglist = [self.share_server.id, '--status', 'active']
        verifylist = [('share_server', self.share_server.id), ('status', 'active')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.servers_mock.reset_state.side_effect = Exception()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)