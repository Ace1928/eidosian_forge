from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
class TestPolicyEnforcer(base.IsolatedUnitTest):

    def test_policy_enforce_unregistered(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[])
        self.assertRaises(glance.api.policy.policy.PolicyNotRegistered, enforcer.enforce, context, 'wibble', {})

    def test_policy_check_unregistered(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[])
        self.assertRaises(glance.api.policy.policy.PolicyNotRegistered, enforcer.check, context, 'wibble', {})

    def test_policy_file_default_rules_default_location(self):
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=['reader'])
        enforcer.enforce(context, 'get_image', {'project_id': context.project_id})

    def test_policy_file_custom_rules_default_location(self):
        rules = {'get_image': '!'}
        self.set_policy_rules(rules)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[])
        self.assertRaises(exception.Forbidden, enforcer.enforce, context, 'get_image', {})

    def test_policy_file_custom_location(self):
        self.config(policy_file=os.path.join(self.test_dir, 'gobble.gobble'), group='oslo_policy')
        rules = {'get_image': '!'}
        self.set_policy_rules(rules)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[])
        self.assertRaises(exception.Forbidden, enforcer.enforce, context, 'get_image', {})

    def test_policy_file_check(self):
        self.config(policy_file=os.path.join(self.test_dir, 'gobble.gobble'), group='oslo_policy')
        rules = {'get_image': '!'}
        self.set_policy_rules(rules)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[])
        self.assertEqual(False, enforcer.check(context, 'get_image', {}))

    def test_policy_file_get_image_default_everybody(self):
        rules = {'default': '', 'get_image': ''}
        self.set_policy_rules(rules)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[])
        self.assertEqual(True, enforcer.check(context, 'get_image', {}))

    def test_policy_file_get_image_default_nobody(self):
        rules = {'default': '!'}
        self.set_policy_rules(rules)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[])
        self.assertRaises(exception.Forbidden, enforcer.enforce, context, 'get_image', {})

    def _test_enforce_scope(self):
        policy_name = 'foo'
        rule = common_policy.RuleDefault(name=policy_name, check_str='role:bar', scope_types=['system'])
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        enforcer.register_default(rule)
        context = glance.context.RequestContext(user_id='user', project_id='project', roles=['bar'])
        target = {}
        return enforcer.enforce(context, policy_name, target)

    def test_policy_enforcer_raises_forbidden_when_enforcing_scope(self):
        self.config(enforce_scope=True, group='oslo_policy')
        self.assertRaises(exception.Forbidden, self._test_enforce_scope)

    def test_policy_enforcer_does_not_raise_forbidden(self):
        self.config(enforce_scope=False, group='oslo_policy')
        self.assertTrue(self._test_enforce_scope())

    def test_ensure_context_object_is_passed_to_policy_enforcement(self):
        context = glance.context.RequestContext()
        mock_enforcer = self.mock_object(common_policy.Enforcer, 'enforce')
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        enforcer.register_default(common_policy.RuleDefault(name='foo', check_str='role:bar'))
        enforcer.enforce(context, 'foo', {})
        mock_enforcer.assert_called_once_with('foo', {}, context, do_raise=True, exc=exception.Forbidden, action='foo')
        mock_enforcer.reset_mock()
        enforcer.check(context, 'foo', {})
        mock_enforcer.assert_called_once_with('foo', {}, context)