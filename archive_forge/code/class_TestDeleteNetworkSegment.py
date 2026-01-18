from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_segment
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteNetworkSegment(TestNetworkSegment):
    _network_segments = network_fakes.create_network_segments()

    def setUp(self):
        super(TestDeleteNetworkSegment, self).setUp()
        self.network_client.delete_segment = mock.Mock(return_value=None)
        self.network_client.find_segment = mock.Mock(side_effect=self._network_segments)
        self.cmd = network_segment.DeleteNetworkSegment(self.app, self.namespace)

    def test_delete(self):
        arglist = [self._network_segments[0].id]
        verifylist = [('network_segment', [self._network_segments[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_segment.assert_called_once_with(self._network_segments[0])
        self.assertIsNone(result)

    def test_delete_multiple(self):
        arglist = []
        for _network_segment in self._network_segments:
            arglist.append(_network_segment.id)
        verifylist = [('network_segment', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for _network_segment in self._network_segments:
            calls.append(call(_network_segment))
        self.network_client.delete_segment.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_with_exception(self):
        arglist = [self._network_segments[0].id, 'doesnotexist']
        verifylist = [('network_segment', [self._network_segments[0].id, 'doesnotexist'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._network_segments[0], exceptions.CommandError]
        self.network_client.find_segment = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 network segments failed to delete.', str(e))
        self.network_client.find_segment.assert_any_call(self._network_segments[0].id, ignore_missing=False)
        self.network_client.find_segment.assert_any_call('doesnotexist', ignore_missing=False)
        self.network_client.delete_segment.assert_called_once_with(self._network_segments[0])