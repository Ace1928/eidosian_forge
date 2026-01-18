from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.network_delete')
class TestDeleteNetworkCompute(compute_fakes.TestComputev2):

    def setUp(self):
        super(TestDeleteNetworkCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self._networks = compute_fakes.create_networks(count=3)
        self.compute_client.api.network_find = compute_fakes.get_networks(networks=self._networks)
        self.cmd = network.DeleteNetwork(self.app, None)

    def test_network_delete_one(self, net_mock):
        net_mock.return_value = mock.Mock(return_value=None)
        arglist = [self._networks[0]['label']]
        verifylist = [('network', [self._networks[0]['label']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        net_mock.assert_called_once_with(self._networks[0]['label'])
        self.assertIsNone(result)

    def test_network_delete_multi(self, net_mock):
        net_mock.return_value = mock.Mock(return_value=None)
        arglist = []
        for n in self._networks:
            arglist.append(n['id'])
        verifylist = [('network', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for n in self._networks:
            calls.append(call(n['id']))
        net_mock.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_network_delete_multi_with_exception(self, net_mock):
        net_mock.return_value = mock.Mock(return_value=None)
        net_mock.side_effect = [mock.Mock(return_value=None), exceptions.CommandError]
        arglist = [self._networks[0]['id'], 'xxxx-yyyy-zzzz', self._networks[1]['id']]
        verifylist = [('network', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('2 of 3 networks failed to delete.', str(e))
        net_mock.assert_any_call(self._networks[0]['id'])
        net_mock.assert_any_call(self._networks[1]['id'])
        net_mock.assert_any_call('xxxx-yyyy-zzzz')