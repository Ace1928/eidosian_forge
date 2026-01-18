import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
class EnforcerCheckRulesTest(base.PolicyBaseTestCase):

    def setUp(self):
        super(EnforcerCheckRulesTest, self).setUp()

    def test_no_violations(self):
        self.create_config_file('policy.json', POLICY_JSON_CONTENTS)
        self.enforcer.load_rules(True)
        self.assertTrue(self.enforcer.check_rules(raise_on_violation=True))

    @mock.patch.object(policy, 'LOG')
    def test_undefined_rule(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar'})
        self.create_config_file('policy.json', rules)
        self.enforcer.load_rules(True)
        self.assertFalse(self.enforcer.check_rules())
        mock_log.warning.assert_called()

    @mock.patch.object(policy, 'LOG')
    def test_undefined_rule_skipped(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar'})
        self.create_config_file('policy.json', rules)
        self.enforcer.skip_undefined_check = True
        self.enforcer.load_rules(True)
        self.assertTrue(self.enforcer.check_rules())

    @mock.patch.object(policy, 'LOG')
    def test_undefined_rule_raises(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar'})
        self.create_config_file('policy.json', rules)
        self.enforcer.load_rules(True)
        self.assertRaises(policy.InvalidDefinitionError, self.enforcer.check_rules, raise_on_violation=True)
        mock_log.warning.assert_called()

    @mock.patch.object(policy, 'LOG')
    def test_undefined_rule_raises_skipped(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar'})
        self.create_config_file('policy.json', rules)
        self.enforcer.skip_undefined_check = True
        self.enforcer.load_rules(True)
        self.assertTrue(self.enforcer.check_rules(raise_on_violation=True))

    @mock.patch.object(policy, 'LOG')
    def test_cyclical_rules(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar', 'bar': 'rule:foo'})
        self.create_config_file('policy.json', rules)
        self.enforcer.load_rules(True)
        self.assertFalse(self.enforcer.check_rules())
        mock_log.warning.assert_called()

    @mock.patch.object(policy, 'LOG')
    def test_cyclical_rules_raises(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar', 'bar': 'rule:foo'})
        self.create_config_file('policy.json', rules)
        self.enforcer.load_rules(True)
        self.assertRaises(policy.InvalidDefinitionError, self.enforcer.check_rules, raise_on_violation=True)
        mock_log.warning.assert_called()

    @mock.patch.object(policy, 'LOG')
    def test_complex_cyclical_rules_false(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar', 'bar': 'rule:baz and role:admin', 'baz': 'rule:foo or role:user'})
        self.create_config_file('policy.json', rules)
        self.enforcer.load_rules(True)
        self.assertFalse(self.enforcer.check_rules())
        mock_log.warning.assert_called()

    def test_complex_cyclical_rules_true(self):
        rules = jsonutils.dumps({'foo': 'rule:bar or rule:baz', 'bar': 'role:admin', 'baz': 'rule:bar or role:user'})
        self.create_config_file('policy.json', rules)
        self.enforcer.load_rules(True)
        self.assertTrue(self.enforcer.check_rules())