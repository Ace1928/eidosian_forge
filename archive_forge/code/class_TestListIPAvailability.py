from unittest import mock
from osc_lib.cli import format_columns
from openstackclient.network.v2 import ip_availability
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListIPAvailability(TestIPAvailability):
    _ip_availability = network_fakes.create_ip_availability(count=3)
    columns = ('Network ID', 'Network Name', 'Total IPs', 'Used IPs')
    data = []
    for net in _ip_availability:
        data.append((net.network_id, net.network_name, net.total_ips, net.used_ips))

    def setUp(self):
        super(TestListIPAvailability, self).setUp()
        self.cmd = ip_availability.ListIPAvailability(self.app, self.namespace)
        self.network_client.network_ip_availabilities = mock.Mock(return_value=self._ip_availability)

    def test_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'ip_version': 4}
        self.network_client.network_ip_availabilities.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_ip_version(self):
        arglist = ['--ip-version', str(4)]
        verifylist = [('ip_version', 4)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'ip_version': 4}
        self.network_client.network_ip_availabilities.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_project(self):
        arglist = ['--project', self.project.name]
        verifylist = [('project', self.project.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': self.project.id, 'ip_version': 4}
        self.network_client.network_ip_availabilities.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))