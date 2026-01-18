from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkSegment(TestNetworkSegment):
    _network_segment = network_fakes.create_one_network_segment()

    def setUp(self):
        super(TestSetNetworkSegment, self).setUp()
        self.network_client.find_segment = mock.Mock(return_value=self._network_segment)
        self.network_client.update_segment = mock.Mock(return_value=self._network_segment)
        self.cmd = network_segment.SetNetworkSegment(self.app, self.namespace)

    def test_set_no_options(self):
        arglist = [self._network_segment.id]
        verifylist = [('network_segment', self._network_segment.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_segment.assert_called_once_with(self._network_segment, **{})
        self.assertIsNone(result)

    def test_set_all_options(self):
        arglist = ['--description', 'new description', '--name', 'new name', self._network_segment.id]
        verifylist = [('description', 'new description'), ('name', 'new name'), ('network_segment', self._network_segment.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'description': 'new description', 'name': 'new name'}
        self.network_client.update_segment.assert_called_once_with(self._network_segment, **attrs)
        self.assertIsNone(result)