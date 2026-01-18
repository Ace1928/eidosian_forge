from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShowShareServer(TestShareServer):

    def setUp(self):
        super(TestShowShareServer, self).setUp()
        self.share_server = manila_fakes.FakeShareServer.create_one_server()
        self.servers_mock.get.return_value = self.share_server
        self.cmd = osc_share_servers.ShowShareServer(self.app, None)
        self.data = tuple(self.share_server._info.values())
        self.columns = tuple(self.share_server._info.keys())

    def test_share_server_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_server_show(self):
        arglist = [self.share_server.id]
        verifylist = [('share_server', self.share_server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.servers_mock.get.assert_called_with(self.share_server.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)