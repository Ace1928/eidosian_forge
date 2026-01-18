from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.security_group_rule_create')
class TestCreateSecurityGroupRuleCompute(compute_fakes.TestComputev2):
    project = identity_fakes.FakeProject.create_one_project()
    domain = identity_fakes.FakeDomain.create_one_domain()
    _security_group_rule = None
    _security_group = compute_fakes.create_one_security_group()

    def _setup_security_group_rule(self, attrs=None):
        self._security_group_rule = compute_fakes.create_one_security_group_rule(attrs)
        expected_columns, expected_data = network_utils.format_security_group_rule_show(self._security_group_rule)
        return (expected_columns, expected_data)

    def setUp(self):
        super(TestCreateSecurityGroupRuleCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.compute_client.api.security_group_find = mock.Mock(return_value=self._security_group)
        self.cmd = security_group_rule.CreateSecurityGroupRule(self.app, None)

    def test_security_group_rule_create_no_options(self, sgr_mock):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_security_group_rule_create_all_remote_options(self, sgr_mock):
        arglist = ['--remote-ip', '10.10.0.0/24', '--remote-group', self._security_group['id'], self._security_group['id']]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_security_group_rule_create_bad_protocol(self, sgr_mock):
        arglist = ['--protocol', 'foo', self._security_group['id']]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_security_group_rule_create_all_protocol_options(self, sgr_mock):
        arglist = ['--protocol', 'tcp', '--proto', 'tcp', self._security_group['id']]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_security_group_rule_create_network_options(self, sgr_mock):
        arglist = ['--ingress', '--ethertype', 'IPv4', '--icmp-type', '3', '--icmp-code', '11', '--project', self.project.name, '--project-domain', self.domain.name, self._security_group['id']]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_security_group_rule_create_default_rule(self, sgr_mock):
        expected_columns, expected_data = self._setup_security_group_rule()
        sgr_mock.return_value = self._security_group_rule
        dst_port = str(self._security_group_rule['from_port']) + ':' + str(self._security_group_rule['to_port'])
        arglist = ['--dst-port', dst_port, self._security_group['id']]
        verifylist = [('dst_port', (self._security_group_rule['from_port'], self._security_group_rule['to_port'])), ('group', self._security_group['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        sgr_mock.assert_called_once_with(security_group_id=self._security_group['id'], ip_protocol=self._security_group_rule['ip_protocol'], from_port=self._security_group_rule['from_port'], to_port=self._security_group_rule['to_port'], remote_ip=self._security_group_rule['ip_range']['cidr'], remote_group=None)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)

    def test_security_group_rule_create_remote_group(self, sgr_mock):
        expected_columns, expected_data = self._setup_security_group_rule({'from_port': 22, 'to_port': 22, 'group': {'name': self._security_group['name']}})
        sgr_mock.return_value = self._security_group_rule
        arglist = ['--dst-port', str(self._security_group_rule['from_port']), '--remote-group', self._security_group['name'], self._security_group['id']]
        verifylist = [('dst_port', (self._security_group_rule['from_port'], self._security_group_rule['to_port'])), ('remote_group', self._security_group['name']), ('group', self._security_group['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        sgr_mock.assert_called_once_with(security_group_id=self._security_group['id'], ip_protocol=self._security_group_rule['ip_protocol'], from_port=self._security_group_rule['from_port'], to_port=self._security_group_rule['to_port'], remote_ip=self._security_group_rule['ip_range']['cidr'], remote_group=self._security_group['id'])
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)

    def test_security_group_rule_create_remote_ip(self, sgr_mock):
        expected_columns, expected_data = self._setup_security_group_rule({'ip_protocol': 'icmp', 'from_port': -1, 'to_port': -1, 'ip_range': {'cidr': '10.0.2.0/24'}})
        sgr_mock.return_value = self._security_group_rule
        arglist = ['--protocol', self._security_group_rule['ip_protocol'], '--remote-ip', self._security_group_rule['ip_range']['cidr'], self._security_group['id']]
        verifylist = [('protocol', self._security_group_rule['ip_protocol']), ('remote_ip', self._security_group_rule['ip_range']['cidr']), ('group', self._security_group['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        sgr_mock.assert_called_once_with(security_group_id=self._security_group['id'], ip_protocol=self._security_group_rule['ip_protocol'], from_port=self._security_group_rule['from_port'], to_port=self._security_group_rule['to_port'], remote_ip=self._security_group_rule['ip_range']['cidr'], remote_group=None)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)

    def test_security_group_rule_create_proto_option(self, sgr_mock):
        expected_columns, expected_data = self._setup_security_group_rule({'ip_protocol': 'icmp', 'from_port': -1, 'to_port': -1, 'ip_range': {'cidr': '10.0.2.0/24'}})
        sgr_mock.return_value = self._security_group_rule
        arglist = ['--proto', self._security_group_rule['ip_protocol'], '--remote-ip', self._security_group_rule['ip_range']['cidr'], self._security_group['id']]
        verifylist = [('proto', self._security_group_rule['ip_protocol']), ('protocol', None), ('remote_ip', self._security_group_rule['ip_range']['cidr']), ('group', self._security_group['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        sgr_mock.assert_called_once_with(security_group_id=self._security_group['id'], ip_protocol=self._security_group_rule['ip_protocol'], from_port=self._security_group_rule['from_port'], to_port=self._security_group_rule['to_port'], remote_ip=self._security_group_rule['ip_range']['cidr'], remote_group=None)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, data)