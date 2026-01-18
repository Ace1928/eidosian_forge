from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateNetworkSegmentRange(TestNetworkSegmentRange):
    _network_segment_range = network_fakes.create_one_network_segment_range()
    columns = ('available', 'default', 'id', 'maximum', 'minimum', 'name', 'network_type', 'physical_network', 'project_id', 'shared', 'used')
    data = (['100-103', '105'], _network_segment_range.default, _network_segment_range.id, _network_segment_range.maximum, _network_segment_range.minimum, _network_segment_range.name, _network_segment_range.network_type, _network_segment_range.physical_network, _network_segment_range.project_id, _network_segment_range.shared, {'3312e4ba67864b2eb53f3f41432f8efc': ['104', '106']})

    def setUp(self):
        super(TestCreateNetworkSegmentRange, self).setUp()
        self.network_client.create_network_segment_range = mock.Mock(return_value=self._network_segment_range)
        self.cmd = network_segment_range.CreateNetworkSegmentRange(self.app, self.namespace)

    def test_create_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_create_invalid_network_type(self):
        arglist = ['--private', '--project', self._network_segment_range.project_id, '--network-type', 'foo', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_create_default_with_project_id(self):
        arglist = ['--project', self._network_segment_range.project_id, '--network-type', 'vxlan', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        verifylist = [('project', self._network_segment_range.project_id), ('network_type', 'vxlan'), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_shared_with_project_id(self):
        arglist = ['--shared', '--project', self._network_segment_range.project_id, '--network-type', 'vxlan', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        verifylist = [('shared', True), ('project', self._network_segment_range.project_id), ('network_type', 'vxlan'), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_tunnel_with_physical_network(self):
        arglist = ['--shared', '--network-type', 'vxlan', '--physical-network', self._network_segment_range.physical_network, '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        verifylist = [('shared', True), ('network_type', 'vxlan'), ('physical_network', self._network_segment_range.physical_network), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_minimum_options(self):
        arglist = ['--network-type', 'vxlan', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        verifylist = [('network_type', 'vxlan'), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network_segment_range.assert_called_once_with(**{'shared': True, 'network_type': 'vxlan', 'minimum': self._network_segment_range.minimum, 'maximum': self._network_segment_range.maximum, 'name': self._network_segment_range.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_private_minimum_options(self):
        arglist = ['--private', '--project', self._network_segment_range.project_id, '--network-type', 'vxlan', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        verifylist = [('shared', False), ('project', self._network_segment_range.project_id), ('network_type', 'vxlan'), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network_segment_range.assert_called_once_with(**{'shared': False, 'project_id': mock.ANY, 'network_type': 'vxlan', 'minimum': self._network_segment_range.minimum, 'maximum': self._network_segment_range.maximum, 'name': self._network_segment_range.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_shared_minimum_options(self):
        arglist = ['--shared', '--network-type', 'vxlan', '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        verifylist = [('shared', True), ('network_type', 'vxlan'), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network_segment_range.assert_called_once_with(**{'shared': True, 'network_type': 'vxlan', 'minimum': self._network_segment_range.minimum, 'maximum': self._network_segment_range.maximum, 'name': self._network_segment_range.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--private', '--project', self._network_segment_range.project_id, '--network-type', self._network_segment_range.network_type, '--physical-network', self._network_segment_range.physical_network, '--minimum', str(self._network_segment_range.minimum), '--maximum', str(self._network_segment_range.maximum), self._network_segment_range.name]
        verifylist = [('shared', self._network_segment_range.shared), ('project', self._network_segment_range.project_id), ('network_type', self._network_segment_range.network_type), ('physical_network', self._network_segment_range.physical_network), ('minimum', self._network_segment_range.minimum), ('maximum', self._network_segment_range.maximum), ('name', self._network_segment_range.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network_segment_range.assert_called_once_with(**{'shared': self._network_segment_range.shared, 'project_id': mock.ANY, 'network_type': self._network_segment_range.network_type, 'physical_network': self._network_segment_range.physical_network, 'minimum': self._network_segment_range.minimum, 'maximum': self._network_segment_range.maximum, 'name': self._network_segment_range.name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)