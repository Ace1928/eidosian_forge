from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.security_group_rule_delete')
class TestDeleteSecurityGroupRuleCompute(compute_fakes.TestComputev2):
    _security_group_rules = compute_fakes.create_security_group_rules(count=2)

    def setUp(self):
        super(TestDeleteSecurityGroupRuleCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = security_group_rule.DeleteSecurityGroupRule(self.app, None)

    def test_security_group_rule_delete(self, sgr_mock):
        arglist = [self._security_group_rules[0]['id']]
        verifylist = [('rule', [self._security_group_rules[0]['id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        sgr_mock.assert_called_once_with(self._security_group_rules[0]['id'])
        self.assertIsNone(result)

    def test_security_group_rule_delete_multi(self, sgr_mock):
        arglist = []
        verifylist = []
        for s in self._security_group_rules:
            arglist.append(s['id'])
        verifylist = [('rule', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for s in self._security_group_rules:
            calls.append(call(s['id']))
        sgr_mock.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_security_group_rule_delete_multi_with_exception(self, sgr_mock):
        arglist = [self._security_group_rules[0]['id'], 'unexist_rule']
        verifylist = [('rule', [self._security_group_rules[0]['id'], 'unexist_rule'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [None, exceptions.CommandError]
        sgr_mock.side_effect = find_mock_result
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 rules failed to delete.', str(e))
        sgr_mock.assert_any_call(self._security_group_rules[0]['id'])
        sgr_mock.assert_any_call('unexist_rule')