from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.floating_ip_delete')
class TestDeleteFloatingIPCompute(compute_fakes.TestComputev2):
    _floating_ips = compute_fakes.create_floating_ips(count=2)

    def setUp(self):
        super(TestDeleteFloatingIPCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = fip.DeleteFloatingIP(self.app, None)

    def test_floating_ip_delete(self, fip_mock):
        fip_mock.return_value = mock.Mock(return_value=None)
        arglist = [self._floating_ips[0]['id']]
        verifylist = [('floating_ip', [self._floating_ips[0]['id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        fip_mock.assert_called_once_with(self._floating_ips[0]['id'])
        self.assertIsNone(result)

    def test_floating_ip_delete_multi(self, fip_mock):
        fip_mock.return_value = mock.Mock(return_value=None)
        arglist = []
        verifylist = []
        for f in self._floating_ips:
            arglist.append(f['id'])
        verifylist = [('floating_ip', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for f in self._floating_ips:
            calls.append(call(f['id']))
        fip_mock.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_floating_ip_delete_multi_exception(self, fip_mock):
        fip_mock.return_value = mock.Mock(return_value=None)
        fip_mock.side_effect = [mock.Mock(return_value=None), exceptions.CommandError]
        arglist = [self._floating_ips[0]['id'], 'unexist_floating_ip']
        verifylist = [('floating_ip', [self._floating_ips[0]['id'], 'unexist_floating_ip'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 floating_ips failed to delete.', str(e))
        fip_mock.assert_any_call(self._floating_ips[0]['id'])
        fip_mock.assert_any_call('unexist_floating_ip')