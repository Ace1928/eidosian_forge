from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteFloatingIPNetwork(TestFloatingIPNetwork):
    floating_ips = network_fakes.FakeFloatingIP.create_floating_ips(count=2)

    def setUp(self):
        super(TestDeleteFloatingIPNetwork, self).setUp()
        self.network_client.delete_ip = mock.Mock(return_value=None)
        self.cmd = fip.DeleteFloatingIP(self.app, self.namespace)

    def test_floating_ip_delete(self):
        self.network_client.find_ip.side_effect = [self.floating_ips[0], self.floating_ips[1]]
        arglist = [self.floating_ips[0].id]
        verifylist = [('floating_ip', [self.floating_ips[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_ip.assert_called_once_with(self.floating_ips[0].id, ignore_missing=False)
        self.network_client.delete_ip.assert_called_once_with(self.floating_ips[0])
        self.assertIsNone(result)

    def test_floating_ip_delete_multi(self):
        self.network_client.find_ip.side_effect = [self.floating_ips[0], self.floating_ips[1]]
        arglist = []
        verifylist = []
        for f in self.floating_ips:
            arglist.append(f.id)
        verifylist = [('floating_ip', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = [call(self.floating_ips[0].id, ignore_missing=False), call(self.floating_ips[1].id, ignore_missing=False)]
        self.network_client.find_ip.assert_has_calls(calls)
        calls = []
        for f in self.floating_ips:
            calls.append(call(f))
        self.network_client.delete_ip.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_floating_ip_delete_multi_exception(self):
        self.network_client.find_ip.side_effect = [self.floating_ips[0], exceptions.CommandError]
        arglist = [self.floating_ips[0].id, 'unexist_floating_ip']
        verifylist = [('floating_ip', [self.floating_ips[0].id, 'unexist_floating_ip'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 floating_ips failed to delete.', str(e))
        self.network_client.find_ip.assert_any_call(self.floating_ips[0].id, ignore_missing=False)
        self.network_client.find_ip.assert_any_call('unexist_floating_ip', ignore_missing=False)
        self.network_client.delete_ip.assert_called_once_with(self.floating_ips[0])