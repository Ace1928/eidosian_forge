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
class TestCreateNetworkIdentityV3(TestNetwork):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    _network = network_fakes.create_one_network(attrs={'project_id': project.id, 'availability_zone_hints': ['nova']})
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy(attrs={'id': _network.qos_policy_id})
    columns = ('admin_state_up', 'availability_zone_hints', 'availability_zones', 'created_at', 'description', 'dns_domain', 'id', 'ipv4_address_scope', 'ipv6_address_scope', 'is_default', 'is_vlan_transparent', 'mtu', 'name', 'port_security_enabled', 'project_id', 'provider:network_type', 'provider:physical_network', 'provider:segmentation_id', 'qos_policy_id', 'router:external', 'shared', 'status', 'segments', 'subnets', 'tags', 'revision_number', 'updated_at')
    data = (network.AdminStateColumn(_network.is_admin_state_up), format_columns.ListColumn(_network.availability_zone_hints), format_columns.ListColumn(_network.availability_zones), _network.created_at, _network.description, _network.dns_domain, _network.id, _network.ipv4_address_scope_id, _network.ipv6_address_scope_id, _network.is_default, _network.mtu, _network.name, _network.is_port_security_enabled, _network.project_id, _network.provider_network_type, _network.provider_physical_network, _network.provider_segmentation_id, _network.qos_policy_id, network.RouterExternalColumn(_network.is_router_external), _network.is_shared, _network.is_vlan_transparent, _network.status, _network.segments, format_columns.ListColumn(_network.subnet_ids), format_columns.ListColumn(_network.tags), _network.revision_number, _network.updated_at)

    def setUp(self):
        super(TestCreateNetworkIdentityV3, self).setUp()
        self.network_client.create_network = mock.Mock(return_value=self._network)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = network.CreateNetwork(self.app, self.namespace)
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.network_client.find_qos_policy = mock.Mock(return_value=self.qos_policy)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = [self._network.name]
        verifylist = [('name', self._network.name), ('enable', True), ('share', None), ('project', None), ('external', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network.assert_called_once_with(**{'admin_state_up': True, 'name': self._network.name})
        self.assertFalse(self.network_client.set_tags.called)
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--disable', '--share', '--description', self._network.description, '--mtu', str(self._network.mtu), '--project', self.project.name, '--project-domain', self.domain.name, '--availability-zone-hint', 'nova', '--external', '--default', '--provider-network-type', 'vlan', '--provider-physical-network', 'physnet1', '--provider-segment', '400', '--qos-policy', self.qos_policy.id, '--transparent-vlan', '--enable-port-security', '--dns-domain', 'example.org.', self._network.name]
        verifylist = [('disable', True), ('share', True), ('description', self._network.description), ('mtu', str(self._network.mtu)), ('project', self.project.name), ('project_domain', self.domain.name), ('availability_zone_hints', ['nova']), ('external', True), ('default', True), ('provider_network_type', 'vlan'), ('physical_network', 'physnet1'), ('segmentation_id', '400'), ('qos_policy', self.qos_policy.id), ('transparent_vlan', True), ('enable_port_security', True), ('name', self._network.name), ('dns_domain', 'example.org.')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network.assert_called_once_with(**{'admin_state_up': False, 'availability_zone_hints': ['nova'], 'name': self._network.name, 'shared': True, 'description': self._network.description, 'mtu': str(self._network.mtu), 'project_id': self.project.id, 'is_default': True, 'router:external': True, 'provider:network_type': 'vlan', 'provider:physical_network': 'physnet1', 'provider:segmentation_id': '400', 'qos_policy_id': self.qos_policy.id, 'vlan_transparent': True, 'port_security_enabled': True, 'dns_domain': 'example.org.'})
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, data)

    def test_create_other_options(self):
        arglist = ['--enable', '--no-share', '--disable-port-security', self._network.name]
        verifylist = [('enable', True), ('no_share', True), ('name', self._network.name), ('external', False), ('disable_port_security', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network.assert_called_once_with(**{'admin_state_up': True, 'name': self._network.name, 'shared': False, 'port_security_enabled': False})
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, data)

    def _test_create_with_tag(self, add_tags=True):
        arglist = [self._network.name]
        if add_tags:
            arglist += ['--tag', 'red', '--tag', 'blue']
        else:
            arglist += ['--no-tag']
        verifylist = [('name', self._network.name), ('enable', True), ('share', None), ('project', None), ('external', False)]
        if add_tags:
            verifylist.append(('tags', ['red', 'blue']))
        else:
            verifylist.append(('no_tag', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network.assert_called_once_with(name=self._network.name, admin_state_up=True)
        if add_tags:
            self.network_client.set_tags.assert_called_once_with(self._network, tests_utils.CompareBySet(['red', 'blue']))
        else:
            self.assertFalse(self.network_client.set_tags.called)
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, data)

    def test_create_with_tags(self):
        self._test_create_with_tag(add_tags=True)

    def test_create_with_no_tag(self):
        self._test_create_with_tag(add_tags=False)