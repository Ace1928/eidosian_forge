from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListSubnet(TestSubnet):
    _subnet = network_fakes.FakeSubnet.create_subnets(count=3)
    columns = ('ID', 'Name', 'Network', 'Subnet')
    columns_long = columns + ('Project', 'DHCP', 'Name Servers', 'Allocation Pools', 'Host Routes', 'IP Version', 'Gateway', 'Service Types', 'Tags')
    data = []
    for subnet in _subnet:
        data.append((subnet.id, subnet.name, subnet.network_id, subnet.cidr))
    data_long = []
    for subnet in _subnet:
        data_long.append((subnet.id, subnet.name, subnet.network_id, subnet.cidr, subnet.project_id, subnet.enable_dhcp, format_columns.ListColumn(subnet.dns_nameservers), subnet_v2.AllocationPoolsColumn(subnet.allocation_pools), subnet_v2.HostRoutesColumn(subnet.host_routes), subnet.ip_version, subnet.gateway_ip, format_columns.ListColumn(subnet.service_types), format_columns.ListColumn(subnet.tags)))

    def setUp(self):
        super(TestListSubnet, self).setUp()
        self.cmd = subnet_v2.ListSubnet(self.app, self.namespace)
        self.network_client.subnets = mock.Mock(return_value=self._subnet)

    def test_subnet_list_no_options(self):
        arglist = []
        verifylist = [('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.subnets.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.subnets.assert_called_once_with()
        self.assertEqual(self.columns_long, columns)
        self.assertCountEqual(self.data_long, list(data))

    def test_subnet_list_ip_version(self):
        arglist = ['--ip-version', str(4)]
        verifylist = [('ip_version', 4)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'ip_version': 4}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_dhcp(self):
        arglist = ['--dhcp']
        verifylist = [('dhcp', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'enable_dhcp': True, 'is_dhcp_enabled': True}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_no_dhcp(self):
        arglist = ['--no-dhcp']
        verifylist = [('no_dhcp', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'enable_dhcp': False, 'is_dhcp_enabled': False}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_service_type(self):
        arglist = ['--service-type', 'network:router_gateway']
        verifylist = [('service_types', ['network:router_gateway'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'service_types': ['network:router_gateway']}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_service_type_multiple(self):
        arglist = ['--service-type', 'network:router_gateway', '--service-type', 'network:floatingip_agent_gateway']
        verifylist = [('service_types', ['network:router_gateway', 'network:floatingip_agent_gateway'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'service_types': ['network:router_gateway', 'network:floatingip_agent_gateway']}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id), ('project_domain', project.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_network(self):
        network = network_fakes.create_one_network()
        self.network_client.find_network = mock.Mock(return_value=network)
        arglist = ['--network', network.id]
        verifylist = [('network', network.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'network_id': network.id}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_gateway(self):
        subnet = network_fakes.FakeSubnet.create_one_subnet()
        self.network_client.find_network = mock.Mock(return_value=subnet)
        arglist = ['--gateway', subnet.gateway_ip]
        verifylist = [('gateway', subnet.gateway_ip)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'gateway_ip': subnet.gateway_ip}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_name(self):
        subnet = network_fakes.FakeSubnet.create_one_subnet()
        self.network_client.find_network = mock.Mock(return_value=subnet)
        arglist = ['--name', subnet.name]
        verifylist = [('name', subnet.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'name': subnet.name}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_subnet_range(self):
        subnet = network_fakes.FakeSubnet.create_one_subnet()
        self.network_client.find_network = mock.Mock(return_value=subnet)
        arglist = ['--subnet-range', subnet.cidr]
        verifylist = [('subnet_range', subnet.cidr)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'cidr': subnet.cidr}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_subnetpool_by_name(self):
        subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool()
        subnet = network_fakes.FakeSubnet.create_one_subnet({'subnetpool_id': subnet_pool.id})
        self.network_client.find_network = mock.Mock(return_value=subnet)
        self.network_client.find_subnet_pool = mock.Mock(return_value=subnet_pool)
        arglist = ['--subnet-pool', subnet_pool.name]
        verifylist = [('subnet_pool', subnet_pool.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'subnetpool_id': subnet_pool.id}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_subnet_list_subnetpool_by_id(self):
        subnet_pool = network_fakes.FakeSubnetPool.create_one_subnet_pool()
        subnet = network_fakes.FakeSubnet.create_one_subnet({'subnetpool_id': subnet_pool.id})
        self.network_client.find_network = mock.Mock(return_value=subnet)
        self.network_client.find_subnet_pool = mock.Mock(return_value=subnet_pool)
        arglist = ['--subnet-pool', subnet_pool.id]
        verifylist = [('subnet_pool', subnet_pool.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'subnetpool_id': subnet_pool.id}
        self.network_client.subnets.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_with_tag_options(self):
        arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
        verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.subnets.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white'})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))