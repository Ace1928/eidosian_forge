from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListSecurityGroupRuleCompute(compute_fakes.TestComputev2):
    _security_group = compute_fakes.create_one_security_group()
    _security_group_rule_tcp = compute_fakes.create_one_security_group_rule({'ip_protocol': 'tcp', 'ethertype': 'IPv4', 'from_port': 80, 'to_port': 80, 'group': {'name': _security_group['name']}})
    _security_group_rule_icmp = compute_fakes.create_one_security_group_rule({'ip_protocol': 'icmp', 'ethertype': 'IPv4', 'from_port': -1, 'to_port': -1, 'ip_range': {'cidr': '10.0.2.0/24'}, 'group': {'name': _security_group['name']}})
    _security_group['rules'] = [_security_group_rule_tcp, _security_group_rule_icmp]
    expected_columns_with_group = ('ID', 'IP Protocol', 'Ethertype', 'IP Range', 'Port Range', 'Direction', 'Remote Security Group')
    expected_columns_no_group = expected_columns_with_group + ('Security Group',)
    expected_data_with_group = []
    expected_data_no_group = []
    for _security_group_rule in _security_group['rules']:
        rule = network_utils.transform_compute_security_group_rule(_security_group_rule)
        expected_rule_with_group = (rule['id'], rule['ip_protocol'], rule['ethertype'], rule['ip_range'], rule['port_range'], rule['remote_security_group'])
        expected_rule_no_group = expected_rule_with_group + (_security_group_rule['parent_group_id'],)
        expected_data_with_group.append(expected_rule_with_group)
        expected_data_no_group.append(expected_rule_no_group)

    def setUp(self):
        super(TestListSecurityGroupRuleCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.compute_client.api.security_group_find = mock.Mock(return_value=self._security_group)
        self.compute_client.api.security_group_list = mock.Mock(return_value=[self._security_group])
        self.cmd = security_group_rule.ListSecurityGroupRule(self.app, None)

    def test_security_group_rule_list_default(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_client.api.security_group_list.assert_called_once_with(search_opts={'all_tenants': False})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))

    def test_security_group_rule_list_with_group(self):
        arglist = [self._security_group['id']]
        verifylist = [('group', self._security_group['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_client.api.security_group_find.assert_called_once_with(self._security_group['id'])
        self.assertEqual(self.expected_columns_with_group, columns)
        self.assertEqual(self.expected_data_with_group, list(data))

    def test_security_group_rule_list_all_projects(self):
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_client.api.security_group_list.assert_called_once_with(search_opts={'all_tenants': True})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))

    def test_security_group_rule_list_with_ignored_options(self):
        arglist = ['--long']
        verifylist = [('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_client.api.security_group_list.assert_called_once_with(search_opts={'all_tenants': False})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))