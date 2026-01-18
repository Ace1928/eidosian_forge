from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment_range
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteNetworkSegmentRange(TestNetworkSegmentRange):
    _network_segment_ranges = network_fakes.create_network_segment_ranges()

    def setUp(self):
        super(TestDeleteNetworkSegmentRange, self).setUp()
        self.network_client.delete_network_segment_range = mock.Mock(return_value=None)
        self.network_client.find_network_segment_range = mock.Mock(side_effect=self._network_segment_ranges)
        self.cmd = network_segment_range.DeleteNetworkSegmentRange(self.app, self.namespace)

    def test_delete(self):
        arglist = [self._network_segment_ranges[0].id]
        verifylist = [('network_segment_range', [self._network_segment_ranges[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_network_segment_range.assert_called_once_with(self._network_segment_ranges[0])
        self.assertIsNone(result)

    def test_delete_multiple(self):
        arglist = []
        for _network_segment_range in self._network_segment_ranges:
            arglist.append(_network_segment_range.id)
        verifylist = [('network_segment_range', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for _network_segment_range in self._network_segment_ranges:
            calls.append(call(_network_segment_range))
        self.network_client.delete_network_segment_range.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_with_exception(self):
        arglist = [self._network_segment_ranges[0].id, 'doesnotexist']
        verifylist = [('network_segment_range', [self._network_segment_ranges[0].id, 'doesnotexist'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._network_segment_ranges[0], exceptions.CommandError]
        self.network_client.find_network_segment_range = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 network segment ranges failed to delete.', str(e))
        self.network_client.find_network_segment_range.assert_any_call(self._network_segment_ranges[0].id, ignore_missing=False)
        self.network_client.find_network_segment_range.assert_any_call('doesnotexist', ignore_missing=False)
        self.network_client.delete_network_segment_range.assert_called_once_with(self._network_segment_ranges[0])