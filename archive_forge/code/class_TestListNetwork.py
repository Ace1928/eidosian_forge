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
class TestListNetwork(TestNetwork):
    _network = network_fakes.create_networks(count=3)
    columns = ('ID', 'Name', 'Subnets')
    columns_long = ('ID', 'Name', 'Status', 'Project', 'State', 'Shared', 'Subnets', 'Network Type', 'Router Type', 'Availability Zones', 'Tags')
    data = []
    for net in _network:
        data.append((net.id, net.name, format_columns.ListColumn(net.subnet_ids)))
    data_long = []
    for net in _network:
        data_long.append((net.id, net.name, net.status, net.project_id, network.AdminStateColumn(net.is_admin_state_up), net.is_shared, format_columns.ListColumn(net.subnet_ids), net.provider_network_type, network.RouterExternalColumn(net.is_router_external), format_columns.ListColumn(net.availability_zones), format_columns.ListColumn(net.tags)))

    def setUp(self):
        super(TestListNetwork, self).setUp()
        self.cmd = network.ListNetwork(self.app, self.namespace)
        self.network_client.networks = mock.Mock(return_value=self._network)
        self._agent = network_fakes.create_one_network_agent()
        self.network_client.get_agent = mock.Mock(return_value=self._agent)
        self.network_client.dhcp_agent_hosting_networks = mock.Mock(return_value=self._network)
        self._tag_list_resource_mock = self.network_client.networks

    def test_network_list_no_options(self):
        arglist = []
        verifylist = [('external', False), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_external(self):
        arglist = ['--external']
        verifylist = [('external', True), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'router:external': True, 'is_router_external': True})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_internal(self):
        arglist = ['--internal']
        verifylist = [('internal', True), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'router:external': False, 'is_router_external': False})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True), ('external', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertCountEqual(self.data_long, list(data))

    def test_list_name(self):
        test_name = 'fakename'
        arglist = ['--name', test_name]
        verifylist = [('external', False), ('long', False), ('name', test_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'name': test_name})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_enable(self):
        arglist = ['--enable']
        verifylist = [('long', False), ('external', False), ('enable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'admin_state_up': True, 'is_admin_state_up': True})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_disable(self):
        arglist = ['--disable']
        verifylist = [('long', False), ('external', False), ('disable', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'admin_state_up': False, 'is_admin_state_up': False})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'project_id': project.id})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.networks.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_share(self):
        arglist = ['--share']
        verifylist = [('long', False), ('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'shared': True, 'is_shared': True})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_no_share(self):
        arglist = ['--no-share']
        verifylist = [('long', False), ('no_share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'shared': False, 'is_shared': False})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_status(self):
        choices = ['ACTIVE', 'BUILD', 'DOWN', 'ERROR']
        test_status = random.choice(choices)
        arglist = ['--status', test_status]
        verifylist = [('long', False), ('status', test_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'status': test_status})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_provider_network_type(self):
        network_type = self._network[0].provider_network_type
        arglist = ['--provider-network-type', network_type]
        verifylist = [('provider_network_type', network_type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'provider:network_type': network_type, 'provider_network_type': network_type})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_provider_physical_network(self):
        physical_network = self._network[0].provider_physical_network
        arglist = ['--provider-physical-network', physical_network]
        verifylist = [('physical_network', physical_network)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'provider:physical_network': physical_network, 'provider_physical_network': physical_network})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_provider_segment(self):
        segmentation_id = self._network[0].provider_segmentation_id
        arglist = ['--provider-segment', segmentation_id]
        verifylist = [('segmentation_id', segmentation_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'provider:segmentation_id': segmentation_id, 'provider_segmentation_id': segmentation_id})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_network_list_dhcp_agent(self):
        arglist = ['--agent', self._agent.id]
        verifylist = [('agent_id', self._agent.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.dhcp_agent_hosting_networks.assert_called_once_with(self._agent)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(list(data), list(self.data))

    def test_list_with_tag_options(self):
        arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
        verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.networks.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white'})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))