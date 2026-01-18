from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNetworkSegmentRange(TestNetworkSegmentRange):
    _network_segment_ranges = network_fakes.create_network_segment_ranges(count=3)
    columns = ('ID', 'Name', 'Default', 'Shared', 'Project ID', 'Network Type', 'Physical Network', 'Minimum ID', 'Maximum ID')
    columns_long = columns + ('Used', 'Available')
    data = []
    for _network_segment_range in _network_segment_ranges:
        data.append((_network_segment_range.id, _network_segment_range.name, _network_segment_range.default, _network_segment_range.shared, _network_segment_range.project_id, _network_segment_range.network_type, _network_segment_range.physical_network, _network_segment_range.minimum, _network_segment_range.maximum))
    data_long = []
    for _network_segment_range in _network_segment_ranges:
        data_long.append((_network_segment_range.id, _network_segment_range.name, _network_segment_range.default, _network_segment_range.shared, _network_segment_range.project_id, _network_segment_range.network_type, _network_segment_range.physical_network, _network_segment_range.minimum, _network_segment_range.maximum, {'3312e4ba67864b2eb53f3f41432f8efc': ['104', '106']}, ['100-103', '105']))

    def setUp(self):
        super(TestListNetworkSegmentRange, self).setUp()
        self.network_client.network_segment_ranges = mock.Mock(return_value=self._network_segment_ranges)
        self.cmd = network_segment_range.ListNetworkSegmentRange(self.app, self.namespace)

    def test_list_no_option(self):
        arglist = []
        verifylist = [('long', False), ('available', False), ('unavailable', False), ('used', False), ('unused', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.network_segment_ranges.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True), ('available', False), ('unavailable', False), ('used', False), ('unused', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.network_segment_ranges.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, list(data))