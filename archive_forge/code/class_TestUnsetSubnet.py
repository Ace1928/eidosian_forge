from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestUnsetSubnet(TestSubnet):

    def setUp(self):
        super(TestUnsetSubnet, self).setUp()
        self._testsubnet = network_fakes.FakeSubnet.create_one_subnet({'dns_nameservers': ['8.8.8.8', '8.8.8.4'], 'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}, {'destination': '10.30.30.30/24', 'nexthop': '10.30.30.1'}], 'allocation_pools': [{'start': '8.8.8.100', 'end': '8.8.8.150'}, {'start': '8.8.8.160', 'end': '8.8.8.170'}], 'service_types': ['network:router_gateway', 'network:floatingip_agent_gateway'], 'gateway_ip': 'fe80::a00a:0:c0de:0:1', 'tags': ['green', 'red']})
        self.network_client.find_subnet = mock.Mock(return_value=self._testsubnet)
        self.network_client.update_subnet = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = subnet_v2.UnsetSubnet(self.app, self.namespace)

    def test_unset_subnet_params(self):
        arglist = ['--dns-nameserver', '8.8.8.8', '--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.1', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.150', '--service-type', 'network:router_gateway', '--gateway', self._testsubnet.name]
        verifylist = [('dns_nameservers', ['8.8.8.8']), ('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.1'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.150'}]), ('service_types', ['network:router_gateway']), ('gateway', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'dns_nameservers': ['8.8.8.4'], 'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}], 'allocation_pools': [{'start': '8.8.8.160', 'end': '8.8.8.170'}], 'service_types': ['network:floatingip_agent_gateway'], 'gateway_ip': None}
        self.network_client.update_subnet.assert_called_once_with(self._testsubnet, **attrs)
        self.assertIsNone(result)

    def test_unset_subnet_wrong_host_routes(self):
        arglist = ['--dns-nameserver', '8.8.8.8', '--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.2', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.150', self._testsubnet.name]
        verifylist = [('dns_nameservers', ['8.8.8.8']), ('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.2'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.150'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_subnet_wrong_allocation_pool(self):
        arglist = ['--dns-nameserver', '8.8.8.8', '--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.1', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.156', self._testsubnet.name]
        verifylist = [('dns_nameservers', ['8.8.8.8']), ('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.1'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.156'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_subnet_wrong_dns_nameservers(self):
        arglist = ['--dns-nameserver', '8.8.8.1', '--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.1', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.150', self._testsubnet.name]
        verifylist = [('dns_nameservers', ['8.8.8.1']), ('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.1'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.150'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_subnet_wrong_service_type(self):
        arglist = ['--dns-nameserver', '8.8.8.8', '--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.1', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.150', '--service-type', 'network:dhcp', self._testsubnet.name]
        verifylist = [('dns_nameservers', ['8.8.8.8']), ('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.1'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.150'}]), ('service_types', ['network:dhcp'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def _test_unset_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['green']
        else:
            arglist = ['--all-tag']
            verifylist = [('all_tag', True)]
            expected_args = []
        arglist.append(self._testsubnet.name)
        verifylist.append(('subnet', self._testsubnet.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_subnet.called)
        self.network_client.set_tags.assert_called_once_with(self._testsubnet, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_unset_with_tags(self):
        self._test_unset_tags(with_tags=True)

    def test_unset_with_all_tag(self):
        self._test_unset_tags(with_tags=False)