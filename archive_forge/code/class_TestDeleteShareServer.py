from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestDeleteShareServer(TestShareServer):

    def setUp(self):
        super(TestDeleteShareServer, self).setUp()
        self.share_server = manila_fakes.FakeShareServer.create_one_server()
        self.servers_mock.get.return_value = self.share_server
        self.cmd = osc_share_servers.DeleteShareServer(self.app, None)

    def test_share_server_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_server_delete(self):
        arglist = [self.share_server.id]
        verifylist = [('share_servers', [self.share_server.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.servers_mock.delete.assert_called_once_with(self.share_server)
        self.assertIsNone(result)

    def test_share_server_delete_wait(self):
        arglist = [self.share_server.id, '--wait']
        verifylist = [('share_servers', [self.share_server.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.servers_mock.delete.assert_called_once_with(self.share_server)
            self.assertIsNone(result)

    def test_share_server_delete_wait_exception(self):
        arglist = [self.share_server.id, '--wait']
        verifylist = [('share_servers', [self.share_server.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)