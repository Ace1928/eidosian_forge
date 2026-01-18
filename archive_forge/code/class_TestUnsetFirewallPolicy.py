import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestUnsetFirewallPolicy(TestFirewallPolicy, common.TestUnsetFWaaS):

    def setUp(self):
        super(TestUnsetFirewallPolicy, self).setUp()
        self.networkclient.update_firewall_policy = mock.Mock(return_value={self.res: _fwp})
        self.mocked = self.networkclient.update_firewall_policy

        def _mock_find_rule(*args, **kwargs):
            return {'id': args[0]}

        def _mock_find_policy(*args, **kwargs):
            return {'id': args[0], 'firewall_rules': _fwp['firewall_rules']}
        self.networkclient.find_firewall_policy.side_effect = _mock_find_policy
        self.networkclient.find_firewall_rule.side_effect = _mock_find_rule
        self.cmd = firewallpolicy.UnsetFirewallPolicy(self.app, self.namespace)

    def test_unset_audited(self):
        target = self.resource['id']
        arglist = [target, '--audited']
        verifylist = [(self.res, target), ('audited', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        body = {'audited': False}
        self.mocked.assert_called_once_with(target, **body)
        self.assertIsNone(result)

    def test_unset_firewall_rule_not_matched(self):
        _fwp['firewall_rules'] = ['old_rule']
        target = self.resource['id']
        rule = 'new_rule'
        arglist = [target, '--firewall-rule', rule]
        verifylist = [(self.res, target), ('firewall_rule', [rule])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        body = {'firewall_rules': _fwp['firewall_rules']}
        self.mocked.assert_called_once_with(target, **body)
        self.assertIsNone(result)

    def test_unset_firewall_rule_matched(self):
        _fwp['firewall_rules'] = ['rule1', 'rule2']
        target = self.resource['id']
        rule = 'rule1'
        arglist = [target, '--firewall-rule', rule]
        verifylist = [(self.res, target), ('firewall_rule', [rule])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        body = {'firewall_rules': ['rule2']}
        self.mocked.assert_called_once_with(target, **body)
        self.assertIsNone(result)
        self.assertEqual(2, self.networkclient.find_firewall_policy.call_count)
        self.assertEqual(1, self.networkclient.find_firewall_rule.call_count)

    def test_unset_all_firewall_rule(self):
        target = self.resource['id']
        arglist = [target, '--all-firewall-rule']
        verifylist = [(self.res, target), ('all_firewall_rule', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        body = {'firewall_rules': []}
        self.mocked.assert_called_once_with(target, **body)
        self.assertIsNone(result)