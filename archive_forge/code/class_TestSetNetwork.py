import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetwork(TestNetwork):
    _network = network_fakes.create_one_network({'tags': ['green', 'red']})
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy(attrs={'id': _network.qos_policy_id})

    def setUp(self):
        super(TestSetNetwork, self).setUp()
        self.network_client.update_network = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.network_client.find_network = mock.Mock(return_value=self._network)
        self.network_client.find_qos_policy = mock.Mock(return_value=self.qos_policy)
        self.cmd = network.SetNetwork(self.app, self.namespace)

    def test_set_this(self):
        arglist = [self._network.name, '--enable', '--name', 'noob', '--share', '--description', self._network.description, '--dns-domain', 'example.org.', '--external', '--default', '--provider-network-type', 'vlan', '--provider-physical-network', 'physnet1', '--provider-segment', '400', '--enable-port-security', '--qos-policy', self.qos_policy.name]
        verifylist = [('network', self._network.name), ('enable', True), ('description', self._network.description), ('name', 'noob'), ('share', True), ('external', True), ('default', True), ('provider_network_type', 'vlan'), ('physical_network', 'physnet1'), ('segmentation_id', '400'), ('enable_port_security', True), ('qos_policy', self.qos_policy.name), ('dns_domain', 'example.org.')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'noob', 'admin_state_up': True, 'description': self._network.description, 'shared': True, 'router:external': True, 'is_default': True, 'provider:network_type': 'vlan', 'provider:physical_network': 'physnet1', 'provider:segmentation_id': '400', 'port_security_enabled': True, 'qos_policy_id': self.qos_policy.id, 'dns_domain': 'example.org.'}
        self.network_client.update_network.assert_called_once_with(self._network, **attrs)
        self.assertIsNone(result)

    def test_set_that(self):
        arglist = [self._network.name, '--disable', '--no-share', '--internal', '--disable-port-security', '--no-qos-policy']
        verifylist = [('network', self._network.name), ('disable', True), ('no_share', True), ('internal', True), ('disable_port_security', True), ('no_qos_policy', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': False, 'shared': False, 'router:external': False, 'port_security_enabled': False, 'qos_policy_id': None}
        self.network_client.update_network.assert_called_once_with(self._network, **attrs)
        self.assertIsNone(result)

    def test_set_to_empty(self):
        arglist = [self._network.name, '--name', '', '--description', '', '--dns-domain', '']
        verifylist = [('network', self._network.name), ('description', ''), ('name', ''), ('dns_domain', '')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': '', 'description': '', 'dns_domain': ''}
        self.network_client.update_network.assert_called_once_with(self._network, **attrs)
        self.assertIsNone(result)

    def test_set_nothing(self):
        arglist = [self._network.name]
        verifylist = [('network', self._network.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_network.called)
        self.assertFalse(self.network_client.set_tags.called)
        self.assertIsNone(result)

    def _test_set_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['red', 'blue', 'green']
        else:
            arglist = ['--no-tag']
            verifylist = [('no_tag', True)]
            expected_args = []
        arglist.append(self._network.name)
        verifylist.append(('network', self._network.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_network.called)
        self.network_client.set_tags.assert_called_once_with(self._network, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_set_with_tags(self):
        self._test_set_tags(with_tags=True)

    def test_set_with_no_tag(self):
        self._test_set_tags(with_tags=False)