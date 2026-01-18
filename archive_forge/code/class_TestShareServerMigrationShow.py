from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareServerMigrationShow(TestShareServer):

    def setUp(self):
        super(TestShareServerMigrationShow, self).setUp()
        self.new_share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.new_share_network
        self.share_server = manila_fakes.FakeShareServer.create_one_server(attrs={'status': 'migrating', 'task_state': 'migration_in_progress'}, methods={'migration_get_progress': None})
        self.servers_mock.get.return_value = self.share_server
        self.cmd = osc_share_servers.ShareServerMigrationShow(self.app, None)

    def test_share_server_migration_show(self):
        arglist = [self.share_server.id]
        verifylist = [('share_server', self.share_server.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.share_server.migration_get_progress.assert_called