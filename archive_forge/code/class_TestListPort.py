from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestListPort(TestPort):
    _ports = network_fakes.create_ports(count=3)
    columns = ('ID', 'Name', 'MAC Address', 'Fixed IP Addresses', 'Status')
    columns_long = ('ID', 'Name', 'MAC Address', 'Fixed IP Addresses', 'Status', 'Security Groups', 'Device Owner', 'Tags')
    data = []
    for prt in _ports:
        data.append((prt.id, prt.name, prt.mac_address, format_columns.ListDictColumn(prt.fixed_ips), prt.status))
    data_long = []
    for prt in _ports:
        data_long.append((prt.id, prt.name, prt.mac_address, format_columns.ListDictColumn(prt.fixed_ips), prt.status, format_columns.ListColumn(prt.security_group_ids), prt.device_owner, format_columns.ListColumn(prt.tags)))

    def setUp(self):
        super(TestListPort, self).setUp()
        self.network_client.ports = mock.Mock(return_value=self._ports)
        fake_router = network_fakes.FakeRouter.create_one_router({'id': 'fake-router-id'})
        fake_network = network_fakes.create_one_network({'id': 'fake-network-id'})
        self.network_client.find_router = mock.Mock(return_value=fake_router)
        self.network_client.find_network = mock.Mock(return_value=fake_network)
        self.app.client_manager.compute = mock.Mock()
        self.compute_client = self.app.client_manager.compute
        self.cmd = port.ListPort(self.app, self.namespace)

    def test_port_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(fields=LIST_FIELDS_TO_RETRIEVE)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_router_opt(self):
        arglist = ['--router', 'fake-router-name']
        verifylist = [('router', 'fake-router-name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'device_id': 'fake-router-id', 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    @mock.patch.object(utils, 'find_resource')
    def test_port_list_with_server_option(self, mock_find):
        fake_server = compute_fakes.create_one_server()
        mock_find.return_value = fake_server
        arglist = ['--server', 'fake-server-name']
        verifylist = [('server', 'fake-server-name')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(device_id=fake_server.id, fields=LIST_FIELDS_TO_RETRIEVE)
        mock_find.assert_called_once_with(mock.ANY, 'fake-server-name')
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_device_id_opt(self):
        arglist = ['--device-id', self._ports[0].device_id]
        verifylist = [('device_id', self._ports[0].device_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'device_id': self._ports[0].device_id, 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_device_owner_opt(self):
        arglist = ['--device-owner', self._ports[0].device_owner]
        verifylist = [('device_owner', self._ports[0].device_owner)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'device_owner': self._ports[0].device_owner, 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_all_opt(self):
        arglist = ['--device-owner', self._ports[0].device_owner, '--router', 'fake-router-name', '--network', 'fake-network-name', '--mac-address', self._ports[0].mac_address]
        verifylist = [('device_owner', self._ports[0].device_owner), ('router', 'fake-router-name'), ('network', 'fake-network-name'), ('mac_address', self._ports[0].mac_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'device_owner': self._ports[0].device_owner, 'device_id': 'fake-router-id', 'network_id': 'fake-network-id', 'mac_address': self._ports[0].mac_address, 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_mac_address_opt(self):
        arglist = ['--mac-address', self._ports[0].mac_address]
        verifylist = [('mac_address', self._ports[0].mac_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'mac_address': self._ports[0].mac_address, 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_fixed_ip_opt_ip_address(self):
        ip_address = self._ports[0].fixed_ips[0]['ip_address']
        arglist = ['--fixed-ip', 'ip-address=%s' % ip_address]
        verifylist = [('fixed_ip', [{'ip-address': ip_address}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'fixed_ips': ['ip_address=%s' % ip_address], 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_fixed_ip_opt_ip_address_substr(self):
        ip_address_ss = self._ports[0].fixed_ips[0]['ip_address'][:-1]
        arglist = ['--fixed-ip', 'ip-substring=%s' % ip_address_ss]
        verifylist = [('fixed_ip', [{'ip-substring': ip_address_ss}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'fixed_ips': ['ip_address_substr=%s' % ip_address_ss], 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_fixed_ip_opt_subnet_id(self):
        subnet_id = self._ports[0].fixed_ips[0]['subnet_id']
        arglist = ['--fixed-ip', 'subnet=%s' % subnet_id]
        verifylist = [('fixed_ip', [{'subnet': subnet_id}])]
        self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet({'id': subnet_id})
        self.network_client.find_subnet = mock.Mock(return_value=self.fake_subnet)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'fixed_ips': ['subnet_id=%s' % subnet_id], 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_fixed_ip_opts(self):
        subnet_id = self._ports[0].fixed_ips[0]['subnet_id']
        ip_address = self._ports[0].fixed_ips[0]['ip_address']
        arglist = ['--fixed-ip', 'subnet=%s,ip-address=%s' % (subnet_id, ip_address)]
        verifylist = [('fixed_ip', [{'subnet': subnet_id, 'ip-address': ip_address}])]
        self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet({'id': subnet_id})
        self.network_client.find_subnet = mock.Mock(return_value=self.fake_subnet)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'fixed_ips': ['subnet_id=%s' % subnet_id, 'ip_address=%s' % ip_address], 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_fixed_ips(self):
        subnet_id = self._ports[0].fixed_ips[0]['subnet_id']
        ip_address = self._ports[0].fixed_ips[0]['ip_address']
        arglist = ['--fixed-ip', 'subnet=%s' % subnet_id, '--fixed-ip', 'ip-address=%s' % ip_address]
        verifylist = [('fixed_ip', [{'subnet': subnet_id}, {'ip-address': ip_address}])]
        self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet({'id': subnet_id, 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.network_client.find_subnet = mock.Mock(return_value=self.fake_subnet)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'fixed_ips': ['subnet_id=%s' % subnet_id, 'ip_address=%s' % ip_address], 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_port_with_long(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(fields=LIST_FIELDS_TO_RETRIEVE + LIST_FIELDS_TO_RETRIEVE_LONG)
        self.assertEqual(self.columns_long, columns)
        self.assertCountEqual(self.data_long, list(data))

    def test_port_list_host(self):
        arglist = ['--host', 'foobar']
        verifylist = [('host', 'foobar')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'binding:host_id': 'foobar', 'fields': LIST_FIELDS_TO_RETRIEVE}
        self.network_client.ports.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_project(self):
        project = identity_fakes.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id, 'fields': LIST_FIELDS_TO_RETRIEVE}
        self.network_client.ports.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_project_domain(self):
        project = identity_fakes.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id), ('project_domain', project.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'project_id': project.id, 'fields': LIST_FIELDS_TO_RETRIEVE}
        self.network_client.ports.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_name(self):
        test_name = 'fakename'
        arglist = ['--name', test_name]
        verifylist = [('name', test_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'name': test_name, 'fields': LIST_FIELDS_TO_RETRIEVE}
        self.network_client.ports.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_list_with_tag_options(self):
        arglist = ['--tags', 'red,blue', '--any-tags', 'red,green', '--not-tags', 'orange,yellow', '--not-any-tags', 'black,white']
        verifylist = [('tags', ['red', 'blue']), ('any_tags', ['red', 'green']), ('not_tags', ['orange', 'yellow']), ('not_any_tags', ['black', 'white'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.ports.assert_called_once_with(**{'tags': 'red,blue', 'any_tags': 'red,green', 'not_tags': 'orange,yellow', 'not_any_tags': 'black,white', 'fields': LIST_FIELDS_TO_RETRIEVE})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_port_list_security_group(self):
        arglist = ['--security-group', 'sg-id1', '--security-group', 'sg-id2']
        verifylist = [('security_groups', ['sg-id1', 'sg-id2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        filters = {'security_group_ids': ['sg-id1', 'sg-id2'], 'fields': LIST_FIELDS_TO_RETRIEVE}
        self.network_client.ports.assert_called_once_with(**filters)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))