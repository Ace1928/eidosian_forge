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
class CheckFunctionTestCase(base.PolicyBaseTestCase):

    def setUp(self):
        super(CheckFunctionTestCase, self).setUp()
        self.create_config_file('policy.json', POLICY_JSON_CONTENTS)

    def test_check_explicit(self):
        rule = base.FakeCheck()
        creds = {}
        result = self.enforcer.enforce(rule, 'target', creds)
        self.assertEqual(('target', creds, self.enforcer), result)

    def test_check_no_rules(self):
        self.create_config_file('policy.json', '{}')
        self.enforcer.default_rule = None
        self.enforcer.load_rules()
        creds = {}
        result = self.enforcer.enforce('rule', 'target', creds)
        self.assertFalse(result)

    def test_check_with_rule(self):
        self.enforcer.set_rules(dict(default=base.FakeCheck()))
        creds = {}
        result = self.enforcer.enforce('default', 'target', creds)
        self.assertEqual(('target', creds, self.enforcer), result)

    def test_check_rule_not_exist_not_empty_policy_file(self):
        self.create_config_file('policy.json', jsonutils.dumps({'a_rule': []}))
        self.enforcer.default_rule = None
        self.enforcer.load_rules()
        creds = {}
        result = self.enforcer.enforce('rule', 'target', creds)
        self.assertFalse(result)

    def test_check_raise_default(self):
        self.enforcer.set_rules(dict(default=_checks.FalseCheck()))
        creds = {}
        self.assertRaisesRegex(policy.PolicyNotAuthorized, ' is disallowed by policy', self.enforcer.enforce, 'rule', 'target', creds, True)

    def test_check_raise_custom_exception(self):
        self.enforcer.set_rules(dict(default=_checks.FalseCheck()))
        creds = {}
        exc = self.assertRaises(MyException, self.enforcer.enforce, 'rule', 'target', creds, True, MyException, 'arg1', 'arg2', kw1='kwarg1', kw2='kwarg2')
        self.assertEqual(('arg1', 'arg2'), exc.args)
        self.assertEqual(dict(kw1='kwarg1', kw2='kwarg2'), exc.kwargs)