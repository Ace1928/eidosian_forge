from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetFloatingIP(TestFloatingIPNetwork):
    floating_network = network_fakes.create_one_network()
    subnet = network_fakes.FakeSubnet.create_one_subnet()
    port = network_fakes.create_one_port()
    floating_ip = network_fakes.FakeFloatingIP.create_one_floating_ip(attrs={'floating_network_id': floating_network.id, 'port_id': port.id, 'tags': ['green', 'red']})

    def setUp(self):
        super(TestSetFloatingIP, self).setUp()
        self.network_client.find_ip = mock.Mock(return_value=self.floating_ip)
        self.network_client.find_port = mock.Mock(return_value=self.port)
        self.network_client.update_ip = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = fip.SetFloatingIP(self.app, self.namespace)

    def test_port_option(self):
        arglist = [self.floating_ip.id, '--port', self.floating_ip.port_id]
        verifylist = [('floating_ip', self.floating_ip.id), ('port', self.floating_ip.port_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        attrs = {'port_id': self.floating_ip.port_id}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)

    def test_fixed_ip_option(self):
        arglist = [self.floating_ip.id, '--port', self.floating_ip.port_id, '--fixed-ip-address', self.floating_ip.fixed_ip_address]
        verifylist = [('floating_ip', self.floating_ip.id), ('port', self.floating_ip.port_id), ('fixed_ip_address', self.floating_ip.fixed_ip_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        attrs = {'port_id': self.floating_ip.port_id, 'fixed_ip_address': self.floating_ip.fixed_ip_address}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)

    def test_description_option(self):
        arglist = [self.floating_ip.id, '--port', self.floating_ip.port_id, '--description', self.floating_ip.description]
        verifylist = [('floating_ip', self.floating_ip.id), ('port', self.floating_ip.port_id), ('description', self.floating_ip.description)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        attrs = {'port_id': self.floating_ip.port_id, 'description': self.floating_ip.description}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)

    def test_qos_policy_option(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        arglist = ['--qos-policy', qos_policy.id, self.floating_ip.id]
        verifylist = [('qos_policy', qos_policy.id), ('floating_ip', self.floating_ip.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        attrs = {'qos_policy_id': qos_policy.id}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)

    def test_port_and_qos_policy_option(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        arglist = ['--qos-policy', qos_policy.id, '--port', self.floating_ip.port_id, self.floating_ip.id]
        verifylist = [('qos_policy', qos_policy.id), ('port', self.floating_ip.port_id), ('floating_ip', self.floating_ip.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        attrs = {'qos_policy_id': qos_policy.id, 'port_id': self.floating_ip.port_id}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)

    def test_no_qos_policy_option(self):
        arglist = ['--no-qos-policy', self.floating_ip.id]
        verifylist = [('no_qos_policy', True), ('floating_ip', self.floating_ip.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        attrs = {'qos_policy_id': None}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)

    def test_port_and_no_qos_policy_option(self):
        arglist = ['--no-qos-policy', '--port', self.floating_ip.port_id, self.floating_ip.id]
        verifylist = [('no_qos_policy', True), ('port', self.floating_ip.port_id), ('floating_ip', self.floating_ip.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        attrs = {'qos_policy_id': None, 'port_id': self.floating_ip.port_id}
        self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
        self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)

    def _test_set_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['red', 'blue', 'green']
        else:
            arglist = ['--no-tag']
            verifylist = [('no_tag', True)]
            expected_args = []
        arglist.extend([self.floating_ip.id])
        verifylist.extend([('floating_ip', self.floating_ip.id)])
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_ip.called)
        self.network_client.set_tags.assert_called_once_with(self.floating_ip, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_set_with_tags(self):
        self._test_set_tags(with_tags=True)

    def test_set_with_no_tag(self):
        self._test_set_tags(with_tags=False)