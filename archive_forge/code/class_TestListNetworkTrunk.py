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
class TestListNetworkTrunk(TestNetworkTrunk):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    new_trunks = network_fakes.create_trunks({'created_at': '2001-01-01 00:00:00', 'updated_at': '2001-01-01 00:00:00'}, count=3)
    columns = ('ID', 'Name', 'Parent Port', 'Description')
    columns_long = columns + ('Status', 'State', 'Created At', 'Updated At')
    data = []
    for t in new_trunks:
        data.append((t['id'], t['name'], t['port_id'], t['description']))
    data_long = []
    for t in new_trunks:
        data_long.append((t['id'], t['name'], t['port_id'], t['description'], t['status'], network_trunk.AdminStateColumn(''), '2001-01-01 00:00:00', '2001-01-01 00:00:00'))

    def setUp(self):
        super().setUp()
        self.network_client.trunks = mock.Mock(return_value=self.new_trunks)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.cmd = network_trunk.ListNetworkTrunk(self.app, self.namespace)

    def test_trunk_list_no_option(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.trunks.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_trunk_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.trunks.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))