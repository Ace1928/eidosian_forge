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
def _test_enforce_scope(self):
    policy_name = 'foo'
    rule = common_policy.RuleDefault(name=policy_name, check_str='role:bar', scope_types=['system'])
    enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    enforcer.register_default(rule)
    context = glance.context.RequestContext(user_id='user', project_id='project', roles=['bar'])
    target = {}
    return enforcer.enforce(context, policy_name, target)