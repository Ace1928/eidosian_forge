import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestShowNetworkTrunk(TestNetworkTrunk):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_trunk = network_fakes.create_one_trunk()
    columns = ('description', 'id', 'is_admin_state_up', 'name', 'port_id', 'project_id', 'status', 'sub_ports', 'tags')
    data = (new_trunk.description, new_trunk.id, new_trunk.is_admin_state_up, new_trunk.name, new_trunk.port_id, new_trunk.project_id, new_trunk.status, format_columns.ListDictColumn(new_trunk.sub_ports), [])

    def setUp(self):
        super().setUp()
        self.network_client.find_trunk = mock.Mock(return_value=self.new_trunk)
        self.network_client.get_trunk = mock.Mock(return_value=self.new_trunk)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.cmd = network_trunk.ShowNetworkTrunk(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self.new_trunk.id]
        verifylist = [('trunk', self.new_trunk.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.get_trunk.assert_called_once_with(self.new_trunk.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)