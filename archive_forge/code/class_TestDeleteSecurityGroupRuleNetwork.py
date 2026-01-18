from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteSecurityGroupRuleNetwork(TestSecurityGroupRuleNetwork):
    _security_group_rules = network_fakes.FakeSecurityGroupRule.create_security_group_rules(count=2)

    def setUp(self):
        super(TestDeleteSecurityGroupRuleNetwork, self).setUp()
        self.network_client.delete_security_group_rule = mock.Mock(return_value=None)
        self.network_client.find_security_group_rule = network_fakes.FakeSecurityGroupRule.get_security_group_rules(self._security_group_rules)
        self.cmd = security_group_rule.DeleteSecurityGroupRule(self.app, self.namespace)

    def test_security_group_rule_delete(self):
        arglist = [self._security_group_rules[0].id]
        verifylist = [('rule', [self._security_group_rules[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_security_group_rule.assert_called_once_with(self._security_group_rules[0])
        self.assertIsNone(result)

    def test_multi_security_group_rules_delete(self):
        arglist = []
        verifylist = []
        for s in self._security_group_rules:
            arglist.append(s.id)
        verifylist = [('rule', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for s in self._security_group_rules:
            calls.append(call(s))
        self.network_client.delete_security_group_rule.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_multi_security_group_rules_delete_with_exception(self):
        arglist = [self._security_group_rules[0].id, 'unexist_rule']
        verifylist = [('rule', [self._security_group_rules[0].id, 'unexist_rule'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self._security_group_rules[0], exceptions.CommandError]
        self.network_client.find_security_group_rule = mock.Mock(side_effect=find_mock_result)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 rules failed to delete.', str(e))
        self.network_client.find_security_group_rule.assert_any_call(self._security_group_rules[0].id, ignore_missing=False)
        self.network_client.find_security_group_rule.assert_any_call('unexist_rule', ignore_missing=False)
        self.network_client.delete_security_group_rule.assert_called_once_with(self._security_group_rules[0])