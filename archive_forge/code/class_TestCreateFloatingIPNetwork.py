from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestCreateFloatingIPNetwork(TestFloatingIPNetwork):
    floating_network = network_fakes.create_one_network()
    subnet = network_fakes.FakeSubnet.create_one_subnet()
    port = network_fakes.create_one_port()
    floating_ip = network_fakes.FakeFloatingIP.create_one_floating_ip(attrs={'floating_network_id': floating_network.id, 'port_id': port.id, 'dns_domain': 'example.org.', 'dns_name': 'fip1'})
    columns = ('description', 'dns_domain', 'dns_name', 'fixed_ip_address', 'floating_ip_address', 'floating_network_id', 'id', 'port_id', 'project_id', 'qos_policy_id', 'router_id', 'status', 'tags')
    data = (floating_ip.description, floating_ip.dns_domain, floating_ip.dns_name, floating_ip.fixed_ip_address, floating_ip.floating_ip_address, floating_ip.floating_network_id, floating_ip.id, floating_ip.port_id, floating_ip.project_id, floating_ip.qos_policy_id, floating_ip.router_id, floating_ip.status, floating_ip.tags)

    def setUp(self):
        super(TestCreateFloatingIPNetwork, self).setUp()
        self.network_client.create_ip = mock.Mock(return_value=self.floating_ip)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.network_client.find_network = mock.Mock(return_value=self.floating_network)
        self.network_client.find_subnet = mock.Mock(return_value=self.subnet)
        self.network_client.find_port = mock.Mock(return_value=self.port)
        self.cmd = fip.CreateFloatingIP(self.app, self.namespace)

    def test_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_default_options(self):
        arglist = [self.floating_ip.floating_network_id]
        verifylist = [('network', self.floating_ip.floating_network_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_ip.assert_called_once_with(**{'floating_network_id': self.floating_ip.floating_network_id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_all_options(self):
        arglist = ['--subnet', self.subnet.id, '--port', self.floating_ip.port_id, '--floating-ip-address', self.floating_ip.floating_ip_address, '--fixed-ip-address', self.floating_ip.fixed_ip_address, '--description', self.floating_ip.description, '--dns-domain', self.floating_ip.dns_domain, '--dns-name', self.floating_ip.dns_name, self.floating_ip.floating_network_id]
        verifylist = [('subnet', self.subnet.id), ('port', self.floating_ip.port_id), ('fixed_ip_address', self.floating_ip.fixed_ip_address), ('network', self.floating_ip.floating_network_id), ('description', self.floating_ip.description), ('dns_domain', self.floating_ip.dns_domain), ('dns_name', self.floating_ip.dns_name), ('floating_ip_address', self.floating_ip.floating_ip_address)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_ip.assert_called_once_with(**{'subnet_id': self.subnet.id, 'port_id': self.floating_ip.port_id, 'floating_ip_address': self.floating_ip.floating_ip_address, 'fixed_ip_address': self.floating_ip.fixed_ip_address, 'floating_network_id': self.floating_ip.floating_network_id, 'description': self.floating_ip.description, 'dns_domain': self.floating_ip.dns_domain, 'dns_name': self.floating_ip.dns_name})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_floating_ip_create_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, self.floating_ip.floating_network_id]
        verifylist = [('network', self.floating_ip.floating_network_id), ('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_ip.assert_called_once_with(**{'floating_network_id': self.floating_ip.floating_network_id, 'project_id': project.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_floating_ip_create_project_domain(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        domain = identity_fakes_v3.FakeDomain.create_one_domain()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.name, '--project-domain', domain.name, self.floating_ip.floating_network_id]
        verifylist = [('network', self.floating_ip.floating_network_id), ('project', project.name), ('project_domain', domain.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_ip.assert_called_once_with(**{'floating_network_id': self.floating_ip.floating_network_id, 'project_id': project.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_floating_ip_with_qos(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        arglist = ['--qos-policy', qos_policy.id, self.floating_ip.floating_network_id]
        verifylist = [('network', self.floating_ip.floating_network_id), ('qos_policy', qos_policy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_ip.assert_called_once_with(**{'floating_network_id': self.floating_ip.floating_network_id, 'qos_policy_id': qos_policy.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def _test_create_with_tag(self, add_tags=True):
        arglist = [self.floating_ip.floating_network_id]
        if add_tags:
            arglist += ['--tag', 'red', '--tag', 'blue']
        else:
            arglist += ['--no-tag']
        verifylist = [('network', self.floating_ip.floating_network_id)]
        if add_tags:
            verifylist.append(('tags', ['red', 'blue']))
        else:
            verifylist.append(('no_tag', True))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.create_ip.assert_called_once_with(**{'floating_network_id': self.floating_ip.floating_network_id})
        if add_tags:
            self.network_client.set_tags.assert_called_once_with(self.floating_ip, tests_utils.CompareBySet(['red', 'blue']))
        else:
            self.assertFalse(self.network_client.set_tags.called)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)

    def test_create_with_tags(self):
        self._test_create_with_tag(add_tags=True)

    def test_create_with_no_tag(self):
        self._test_create_with_tag(add_tags=False)