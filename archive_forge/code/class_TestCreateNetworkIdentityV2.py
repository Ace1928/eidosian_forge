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
class TestCreateNetworkIdentityV2(TestNetwork):
    project = identity_fakes_v2.FakeProject.create_one_project()
    _network = network_fakes.create_one_network(attrs={'project_id': project.id})
    columns = ('admin_state_up', 'availability_zone_hints', 'availability_zones', 'created_at', 'description', 'dns_domain', 'id', 'ipv4_address_scope', 'ipv6_address_scope', 'is_default', 'is_vlan_transparent', 'mtu', 'name', 'port_security_enabled', 'project_id', 'provider:network_type', 'provider:physical_network', 'provider:segmentation_id', 'qos_policy_id', 'router:external', 'shared', 'status', 'segments', 'subnets', 'tags', 'revision_number', 'updated_at')
    data = (network.AdminStateColumn(_network.is_admin_state_up), format_columns.ListColumn(_network.availability_zone_hints), format_columns.ListColumn(_network.availability_zones), _network.created_at, _network.description, _network.dns_domain, _network.id, _network.ipv4_address_scope_id, _network.ipv6_address_scope_id, _network.is_default, _network.mtu, _network.name, _network.is_port_security_enabled, _network.project_id, _network.provider_network_type, _network.provider_physical_network, _network.provider_segmentation_id, _network.qos_policy_id, network.RouterExternalColumn(_network.is_router_external), _network.is_shared, _network.is_vlan_transparent, _network.status, _network.segments, format_columns.ListColumn(_network.subnet_ids), format_columns.ListColumn(_network.tags), _network.revision_number, _network.updated_at)

    def setUp(self):
        super(TestCreateNetworkIdentityV2, self).setUp()
        self.network_client.create_network = mock.Mock(return_value=self._network)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = network.CreateNetwork(self.app, self.namespace)
        identity_client = identity_fakes_v2.FakeIdentityv2Client(endpoint=fakes.AUTH_URL, token=fakes.AUTH_TOKEN)
        self.app.client_manager.identity = identity_client
        self.identity = self.app.client_manager.identity
        self.projects_mock = self.identity.tenants
        self.projects_mock.get.return_value = self.project

    def test_create_with_project_identityv2(self):
        arglist = ['--project', self.project.name, self._network.name]
        verifylist = [('enable', True), ('share', None), ('name', self._network.name), ('project', self.project.name), ('external', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_network.assert_called_once_with(**{'admin_state_up': True, 'name': self._network.name, 'project_id': self.project.id})
        self.assertFalse(self.network_client.set_tags.called)
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, data)

    def test_create_with_domain_identityv2(self):
        arglist = ['--project', self.project.name, '--project-domain', 'domain-name', self._network.name]
        verifylist = [('enable', True), ('share', None), ('project', self.project.name), ('project_domain', 'domain-name'), ('name', self._network.name), ('external', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(AttributeError, self.cmd.take_action, parsed_args)