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
class DocumentedRuleDefaultDeprecationTestCase(base.PolicyBaseTestCase):

    @mock.patch('warnings.warn', new=mock.Mock())
    def test_deprecate_a_policy_check_string(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(rule_list)
        expected_msg = 'Policy "foo:create_bar":"role:fizz" was deprecated in N in favor of "foo:create_bar":"role:bang". Reason: "role:bang" is a better default. Either ensure your deployment is ready for the new default or copy/paste the deprecated policy into your policy file and maintain it manually.'
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_called_once_with(expected_msg)
        self.assertTrue(enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
        self.assertTrue(enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
        self.assertFalse(enforcer.enforce('foo:create_bar', {}, {'roles': ['baz']}))

    @mock.patch('warnings.warn', new=mock.Mock())
    def test_deprecate_an_empty_policy_check_string(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='', deprecated_reason='because of reasons', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_called_once()
        enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}, do_raise=True)
        enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}, do_raise=True)

    @mock.patch('warnings.warn', new=mock.Mock())
    def test_deprecate_replace_with_empty_policy_check_string(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz', deprecated_reason='because of reasons', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_called_once()
        enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}, do_raise=True)
        enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}, do_raise=True)

    def test_deprecate_a_policy_name(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:baz', deprecated_reason='"foo:bar" is not granular enough. If your deployment has overridden "foo:bar", ensure you override the new policies with same role or rule. Not doing this will require the service to assume the new defaults for "foo:bar:create", "foo:bar:update", "foo:bar:list", and "foo:bar:delete", which might be backwards incompatible for your deployment', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:baz', description='Create a bar.', operations=[{'path': '/v1/bars/', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        expected_msg = 'Policy "foo:bar":"role:baz" was deprecated in N in favor of "foo:create_bar":"role:baz". Reason: "foo:bar" is not granular enough. If your deployment has overridden "foo:bar", ensure you override the new policies with same role or rule. Not doing this will require the service to assume the new defaults for "foo:bar:create", "foo:bar:update", "foo:bar:list", and "foo:bar:delete", which might be backwards incompatible for your deployment. Either ensure your deployment is ready for the new default or copy/paste the deprecated policy into your policy file and maintain it manually.'
        rules = jsonutils.dumps({'foo:bar': 'role:bang'})
        self.create_config_file('policy.json', rules)
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules(True)
            mock_warn.assert_called_once_with(expected_msg)

    def test_deprecate_a_policy_for_removal_logs_warning_when_overridden(self):
        rule_list = [policy.DocumentedRuleDefault(name='foo:bar', check_str='role:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_for_removal=True, deprecated_reason='"foo:bar" is no longer a policy used by the service', deprecated_since='N')]
        expected_msg = 'Policy "foo:bar":"role:baz" was deprecated for removal in N. Reason: "foo:bar" is no longer a policy used by the service. Its value may be silently ignored in the future.'
        rules = jsonutils.dumps({'foo:bar': 'role:bang'})
        self.create_config_file('policy.json', rules)
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_called_once_with(expected_msg)

    def test_deprecate_a_policy_for_removal_does_not_log_warning(self):
        rule_list = [policy.DocumentedRuleDefault(name='foo:bar', check_str='role:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_for_removal=True, deprecated_reason='"foo:bar" is no longer a policy used by the service', deprecated_since='N')]
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_not_called()

    def test_deprecate_check_str_suppress_does_not_log_warning(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        enforcer = policy.Enforcer(self.conf)
        enforcer.suppress_deprecation_warnings = True
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_not_called()

    def test_deprecate_name_suppress_does_not_log_warning(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:baz', deprecated_reason='"foo:bar" is not granular enough.', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:baz', description='Create a bar.', operations=[{'path': '/v1/bars/', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        rules = jsonutils.dumps({'foo:bar': 'role:bang'})
        self.create_config_file('policy.json', rules)
        enforcer = policy.Enforcer(self.conf)
        enforcer.suppress_deprecation_warnings = True
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_not_called()

    def test_deprecate_for_removal_suppress_does_not_log_warning(self):
        rule_list = [policy.DocumentedRuleDefault(name='foo:bar', check_str='role:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_for_removal=True, deprecated_reason='"foo:bar" is no longer a policy used by the service', deprecated_since='N')]
        rules = jsonutils.dumps({'foo:bar': 'role:bang'})
        self.create_config_file('policy.json', rules)
        enforcer = policy.Enforcer(self.conf)
        enforcer.suppress_deprecation_warnings = True
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_not_called()

    def test_suppress_default_change_warnings_flag_not_log_warning(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        enforcer = policy.Enforcer(self.conf)
        enforcer.suppress_default_change_warnings = True
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_not_called()

    def test_deprecated_policy_for_removal_must_include_deprecated_meta(self):
        self.assertRaises(ValueError, policy.DocumentedRuleDefault, name='foo:bar', check_str='rule:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_for_removal=True, deprecated_reason='Some reason.')

    def test_deprecated_policy_should_not_include_deprecated_meta(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='rule:baz')
        with mock.patch('warnings.warn') as mock_warn:
            policy.DocumentedRuleDefault(name='foo:bar', check_str='rule:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_rule=deprecated_rule, deprecated_reason='Some reason.')
            mock_warn.assert_called_once()

    def test_deprecated_rule_requires_deprecated_rule_object(self):
        self.assertRaises(ValueError, policy.DocumentedRuleDefault, name='foo:bar', check_str='rule:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_rule='foo:bar', deprecated_reason='Some reason.')

    def test_deprecated_policy_must_include_deprecated_reason(self):
        self.assertRaises(ValueError, policy.DocumentedRuleDefault, name='foo:bar', check_str='rule:baz', description='Create a foo.', operations=[{'path': '/v1/foos/', 'method': 'POST'}], deprecated_for_removal=True, deprecated_since='N')

    @mock.patch('warnings.warn', new=mock.Mock())
    def test_override_deprecated_policy_with_old_name(self):
        rules = jsonutils.dumps({'foo:bar': 'role:bazz'})
        self.create_config_file('policy.json', rules)
        deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        self.enforcer.register_defaults(rule_list)
        self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
        self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
        self.assertTrue(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bazz']}))

    def test_override_deprecated_policy_with_new_name(self):
        rules = jsonutils.dumps({'foo:create_bar': 'role:bazz'})
        self.create_config_file('policy.json', rules)
        deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        self.enforcer.register_defaults(rule_list)
        self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
        self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
        self.assertTrue(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bazz']}))

    @mock.patch('warnings.warn', new=mock.Mock())
    def test_override_both_new_and_old_policy(self):
        rules_dict = {'foo:create_bar': 'role:bazz', 'foo:bar': 'role:wee'}
        rules = jsonutils.dumps(rules_dict)
        self.create_config_file('policy.json', rules)
        deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        self.enforcer.register_defaults(rule_list)
        self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
        self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
        self.assertFalse(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['wee']}))
        self.assertTrue(self.enforcer.enforce('foo:create_bar', {}, {'roles': ['bazz']}))

    @mock.patch('warnings.warn', new=mock.Mock())
    def test_override_deprecated_policy_with_new_rule(self):
        rules = jsonutils.dumps({'old_rule': 'rule:new_rule'})
        self.create_config_file('policy.json', rules)
        deprecated_rule = policy.DeprecatedRule(name='old_rule', check_str='role:bang', deprecated_reason='"old_rule" is a bad name', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='new_rule', check_str='role:bang', description='Replacement for old_rule.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        self.enforcer.register_defaults(rule_list)
        self.assertFalse(self.enforcer.enforce('new_rule', {}, {'roles': ['fizz']}))
        self.assertTrue(self.enforcer.enforce('new_rule', {}, {'roles': ['bang']}))
        self.assertEqual('bang', self.enforcer.rules['new_rule'].match)

    def test_enforce_new_defaults_no_old_check_string(self):
        self.conf.set_override('enforce_new_defaults', True, group='oslo_policy')
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz', deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults(rule_list)
        with mock.patch('warnings.warn') as mock_warn:
            enforcer.load_rules()
            mock_warn.assert_not_called()
        self.assertTrue(enforcer.enforce('foo:create_bar', {}, {'roles': ['bang']}))
        self.assertFalse(enforcer.enforce('foo:create_bar', {}, {'roles': ['fizz']}))
        self.assertFalse(enforcer.enforce('foo:create_bar', {}, {'roles': ['baz']}))

    def test_deprecation_logic_is_only_performed_once_per_rule(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:fizz')
        rule = policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:bang', description='Create a bar.', operations=[{'path': '/v1/bars', 'method': 'POST'}], deprecated_rule=deprecated_rule, deprecated_reason='"role:bang" is a better default', deprecated_since='N')
        check = rule.check
        enforcer = policy.Enforcer(self.conf)
        enforcer.register_defaults([rule])
        self.assertEqual({}, enforcer.rules)
        enforcer.load_rules()
        expected_check = policy.OrCheck([_parser.parse_rule(cs) for cs in [rule.check_str, deprecated_rule.check_str]])
        self.assertIn('foo:create_bar', enforcer.rules)
        self.assertEqual(str(enforcer.rules['foo:create_bar']), str(expected_check))
        self.assertEqual(check, rule.check)
        enforcer.rules['foo:create_bar'] = 'foo:bar'
        enforcer.load_rules()
        self.assertEqual('foo:bar', enforcer.rules['foo:create_bar'])