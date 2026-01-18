from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowSecurityGroupRuleCompute(compute_fakes.TestComputev2):
    _security_group_rule = compute_fakes.create_one_security_group_rule()
    columns, data = network_utils.format_security_group_rule_show(_security_group_rule)

    def setUp(self):
        super(TestShowSecurityGroupRuleCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        security_group_rules = [self._security_group_rule]
        security_group = {'rules': security_group_rules}
        self.compute_client.api.security_group_list = mock.Mock(return_value=[security_group])
        self.cmd = security_group_rule.ShowSecurityGroupRule(self.app, None)

    def test_security_group_rule_show_no_options(self):
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, [], [])

    def test_security_group_rule_show_all_options(self):
        arglist = [self._security_group_rule['id']]
        verifylist = [('rule', self._security_group_rule['id'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_client.api.security_group_list.assert_called_once_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)