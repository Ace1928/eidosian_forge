from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareServerMigrationComplete(TestShareServer):

    def setUp(self):
        super(TestShareServerMigrationComplete, self).setUp()
        self.share_server = manila_fakes.FakeShareServer.create_one_server(attrs={'status': 'migrating'}, methods={'migration_complete': None})
        self.servers_mock.get.return_value = self.share_server
        self.cmd = osc_share_servers.ShareServerMigrationComplete(self.app, None)

    def test_share_server_migration_complete(self):
        arglist = [self.share_server.id]
        verifylist = [('share_server', self.share_server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_server.migration_complete.assert_called