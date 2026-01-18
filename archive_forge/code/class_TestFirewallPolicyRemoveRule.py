import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallpolicy
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
class TestFirewallPolicyRemoveRule(TestFirewallPolicy):

    def setUp(self):
        super(TestFirewallPolicyRemoveRule, self).setUp()
        self.networkclient.remove_rule_firewall_policy = mock.Mock(return_value={self.res: _fwp})
        self.mocked = self.networkclient.remove_rule_from_policy

        def _mock_find_policy(*args, **kwargs):
            return {'id': args[0]}
        self.networkclient.find_firewall_policy.side_effect = _mock_find_policy
        self.networkclient.find_firewall_rule.side_effect = _mock_find_policy
        self.cmd = firewallpolicy.FirewallPolicyRemoveRule(self.app, self.namespace)

    def test_remove_firewall_rule(self):
        target = self.resource['id']
        rule = 'remove-rule'
        arglist = [target, rule]
        verifylist = [(self.res, target), ('firewall_rule', rule)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.mocked.assert_called_once_with(target, {'firewall_rule_id': rule})
        self.assertIsNone(result)
        self.assertEqual(1, self.networkclient.find_firewall_policy.call_count)
        self.assertEqual(1, self.networkclient.find_firewall_rule.call_count)

    def test_remove_with_no_firewall_rule(self):
        target = self.resource['id']
        arglist = [target]
        verifylist = [(self.res, target)]
        self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)