from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateNetworkSegment(TestNetworkSegment):
    _network_segment = network_fakes.create_one_network_segment()
    _network = network_fakes.create_one_network({'id': _network_segment.network_id})
    columns = ('description', 'id', 'name', 'network_id', 'network_type', 'physical_network', 'segmentation_id')
    data = (_network_segment.description, _network_segment.id, _network_segment.name, _network_segment.network_id, _network_segment.network_type, _network_segment.physical_network, _network_segment.segmentation_id)

    def setUp(self):
        super(TestCreateNetworkSegment, self).setUp()
        self.network_client.create_segment = mock.Mock(return_value=self._network_segment)
        self.network_client.find_network = mock.Mock(return_value=self._network)
        self.cmd = network_segment.CreateNetworkSegment(self.app, self.namespace)

    def test_create_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_create_invalid_network_type(self):
        arglist = ['--network', self._network_segment.network_id, '--network-type', 'foo', self._network_segment.name]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_create_minimum_options(self):
        arglist = ['--network', self._network_segment.network_id, '--network-type', self._network_segment.network_type, self._network_segment.name]
        verifylist = [('network', self._network_segment.network_id), ('network_type', self._network_segment.network_type), ('name', self._network_segment.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_network.assert_called_once_with(self._network_segment.network_id, ignore_missing=False)
        self.network_client.create_segment.assert_called_once_with(**{'network_id': self._network_segment.network_id, 'network_type': self._network_segment.network_type, 'name': self._network_segment.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--description', self._network_segment.description, '--network', self._network_segment.network_id, '--network-type', self._network_segment.network_type, '--physical-network', self._network_segment.physical_network, '--segment', str(self._network_segment.segmentation_id), self._network_segment.name]
        verifylist = [('description', self._network_segment.description), ('network', self._network_segment.network_id), ('network_type', self._network_segment.network_type), ('physical_network', self._network_segment.physical_network), ('segment', self._network_segment.segmentation_id), ('name', self._network_segment.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_network.assert_called_once_with(self._network_segment.network_id, ignore_missing=False)
        self.network_client.create_segment.assert_called_once_with(**{'description': self._network_segment.description, 'network_id': self._network_segment.network_id, 'network_type': self._network_segment.network_type, 'physical_network': self._network_segment.physical_network, 'segmentation_id': self._network_segment.segmentation_id, 'name': self._network_segment.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)