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
class RuleDefaultTestCase(base.PolicyBaseTestCase):

    def test_rule_is_parsed(self):
        opt = policy.RuleDefault(name='foo', check_str='rule:foo')
        self.assertIsInstance(opt.check, _checks.BaseCheck)
        self.assertEqual('rule:foo', str(opt.check))

    def test_str(self):
        opt = policy.RuleDefault(name='foo', check_str='rule:foo')
        self.assertEqual('"foo": "rule:foo"', str(opt))

    def test_equality_obvious(self):
        opt1 = policy.RuleDefault(name='foo', check_str='rule:foo', description='foo')
        opt2 = policy.RuleDefault(name='foo', check_str='rule:foo', description='bar')
        self.assertEqual(opt1, opt2)

    def test_equality_less_obvious(self):
        opt1 = policy.RuleDefault(name='foo', check_str='', description='foo')
        opt2 = policy.RuleDefault(name='foo', check_str='@', description='bar')
        self.assertEqual(opt1, opt2)

    def test_not_equal_check(self):
        opt1 = policy.RuleDefault(name='foo', check_str='rule:foo', description='foo')
        opt2 = policy.RuleDefault(name='foo', check_str='rule:bar', description='bar')
        self.assertNotEqual(opt1, opt2)

    def test_not_equal_name(self):
        opt1 = policy.RuleDefault(name='foo', check_str='rule:foo', description='foo')
        opt2 = policy.RuleDefault(name='bar', check_str='rule:foo', description='bar')
        self.assertNotEqual(opt1, opt2)

    def test_not_equal_class(self):

        class NotRuleDefault(object):

            def __init__(self, name, check_str):
                self.name = name
                self.check = _parser.parse_rule(check_str)
        opt1 = policy.RuleDefault(name='foo', check_str='rule:foo')
        opt2 = NotRuleDefault(name='foo', check_str='rule:foo')
        self.assertNotEqual(opt1, opt2)

    def test_equal_subclass(self):

        class RuleDefaultSub(policy.RuleDefault):
            pass
        opt1 = policy.RuleDefault(name='foo', check_str='rule:foo')
        opt2 = RuleDefaultSub(name='foo', check_str='rule:foo')
        self.assertEqual(opt1, opt2)

    def test_not_equal_subclass(self):

        class RuleDefaultSub(policy.RuleDefault):
            pass
        opt1 = policy.RuleDefault(name='foo', check_str='rule:foo')
        opt2 = RuleDefaultSub(name='bar', check_str='rule:foo')
        self.assertNotEqual(opt1, opt2)

    def test_create_opt_with_scope_types(self):
        scope_types = ['project']
        opt = policy.RuleDefault(name='foo', check_str='role:bar', scope_types=scope_types)
        self.assertEqual(opt.scope_types, scope_types)

    def test_create_opt_with_scope_type_strings_fails(self):
        self.assertRaises(ValueError, policy.RuleDefault, name='foo', check_str='role:bar', scope_types='project')

    def test_create_opt_with_multiple_scope_types(self):
        opt = policy.RuleDefault(name='foo', check_str='role:bar', scope_types=['project', 'domain', 'system'])
        self.assertEqual(opt.scope_types, ['project', 'domain', 'system'])

    def test_ensure_scope_types_are_unique(self):
        self.assertRaises(ValueError, policy.RuleDefault, name='foo', check_str='role:bar', scope_types=['project', 'project'])