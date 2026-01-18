from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkSegment(TestNetworkSegment):
    _network_segment = network_fakes.create_one_network_segment()
    columns = ('description', 'id', 'name', 'network_id', 'network_type', 'physical_network', 'segmentation_id')
    data = (_network_segment.description, _network_segment.id, _network_segment.name, _network_segment.network_id, _network_segment.network_type, _network_segment.physical_network, _network_segment.segmentation_id)

    def setUp(self):
        super(TestShowNetworkSegment, self).setUp()
        self.network_client.find_segment = mock.Mock(return_value=self._network_segment)
        self.cmd = network_segment.ShowNetworkSegment(self.app, self.namespace)

    def test_show_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_show_all_options(self):
        arglist = [self._network_segment.id]
        verifylist = [('network_segment', self._network_segment.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_segment.assert_called_once_with(self._network_segment.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)