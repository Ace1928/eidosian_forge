from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListSecurityGroupRuleNetwork(TestSecurityGroupRuleNetwork):
    _security_group = network_fakes.FakeSecurityGroup.create_one_security_group()
    _security_group_rule_tcp = network_fakes.FakeSecurityGroupRule.create_one_security_group_rule({'protocol': 'tcp', 'port_range_max': 80, 'port_range_min': 80, 'security_group_id': _security_group.id})
    _security_group_rule_icmp = network_fakes.FakeSecurityGroupRule.create_one_security_group_rule({'protocol': 'icmp', 'remote_ip_prefix': '10.0.2.0/24', 'security_group_id': _security_group.id})
    _security_group.security_group_rules = [_security_group_rule_tcp._info, _security_group_rule_icmp._info]
    _security_group_rules = [_security_group_rule_tcp, _security_group_rule_icmp]
    expected_columns_with_group = ('ID', 'IP Protocol', 'Ethertype', 'IP Range', 'Port Range', 'Direction', 'Remote Security Group', 'Remote Address Group')
    expected_columns_no_group = ('ID', 'IP Protocol', 'Ethertype', 'IP Range', 'Port Range', 'Direction', 'Remote Security Group', 'Remote Address Group', 'Security Group')
    expected_data_with_group = []
    expected_data_no_group = []
    for _security_group_rule in _security_group_rules:
        expected_data_with_group.append((_security_group_rule.id, _security_group_rule.protocol, _security_group_rule.ether_type, _security_group_rule.remote_ip_prefix, network_utils.format_network_port_range(_security_group_rule), _security_group_rule.direction, _security_group_rule.remote_group_id, _security_group_rule.remote_address_group_id))
        expected_data_no_group.append((_security_group_rule.id, _security_group_rule.protocol, _security_group_rule.ether_type, _security_group_rule.remote_ip_prefix, network_utils.format_network_port_range(_security_group_rule), _security_group_rule.direction, _security_group_rule.remote_group_id, _security_group_rule.remote_address_group_id, _security_group_rule.security_group_id))

    def setUp(self):
        super(TestListSecurityGroupRuleNetwork, self).setUp()
        self.network_client.find_security_group = mock.Mock(return_value=self._security_group)
        self.network_client.security_group_rules = mock.Mock(return_value=self._security_group_rules)
        self.cmd = security_group_rule.ListSecurityGroupRule(self.app, self.namespace)

    def test_list_default(self):
        self._security_group_rule_tcp.port_range_min = 80
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_group_rules.assert_called_once_with(**{})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))

    def test_list_with_group(self):
        self._security_group_rule_tcp.port_range_min = 80
        arglist = [self._security_group.id]
        verifylist = [('group', self._security_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_group_rules.assert_called_once_with(**{'security_group_id': self._security_group.id})
        self.assertEqual(self.expected_columns_with_group, columns)
        self.assertEqual(self.expected_data_with_group, list(data))

    def test_list_with_ignored_options(self):
        self._security_group_rule_tcp.port_range_min = 80
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_group_rules.assert_called_once_with(**{})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))

    def test_list_with_protocol(self):
        self._security_group_rule_tcp.port_range_min = 80
        arglist = ['--protocol', 'tcp']
        verifylist = [('protocol', 'tcp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_group_rules.assert_called_once_with(**{'protocol': 'tcp'})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))

    def test_list_with_ingress(self):
        self._security_group_rule_tcp.port_range_min = 80
        arglist = ['--ingress']
        verifylist = [('ingress', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_group_rules.assert_called_once_with(**{'direction': 'ingress'})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))

    def test_list_with_wrong_egress(self):
        self._security_group_rule_tcp.port_range_min = 80
        arglist = ['--egress']
        verifylist = [('egress', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.security_group_rules.assert_called_once_with(**{'direction': 'egress'})
        self.assertEqual(self.expected_columns_no_group, columns)
        self.assertEqual(self.expected_data_no_group, list(data))