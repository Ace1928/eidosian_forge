from unittest import mock
from osc_lib.cli import format_columns
from openstackclient.network.v2 import ip_availability
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowIPAvailability(TestIPAvailability):
    _network = network_fakes.create_one_network()
    _ip_availability = network_fakes.create_one_ip_availability(attrs={'network_id': _network.id})
    columns = ('network_id', 'network_name', 'project_id', 'subnet_ip_availability', 'total_ips', 'used_ips')
    data = (_ip_availability.network_id, _ip_availability.network_name, _ip_availability.project_id, format_columns.ListDictColumn(_ip_availability.subnet_ip_availability), _ip_availability.total_ips, _ip_availability.used_ips)

    def setUp(self):
        super(TestShowIPAvailability, self).setUp()
        self.network_client.find_network_ip_availability = mock.Mock(return_value=self._ip_availability)
        self.network_client.find_network = mock.Mock(return_value=self._network)
        self.cmd = ip_availability.ShowIPAvailability(self.app, self.namespace)

    def test_show_no_option(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._ip_availability.network_name]
        verifylist = [('network', self._ip_availability.network_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_network_ip_availability.assert_called_once_with(self._ip_availability.network_id, ignore_missing=False)
        self.network_client.find_network.assert_called_once_with(self._ip_availability.network_name, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)