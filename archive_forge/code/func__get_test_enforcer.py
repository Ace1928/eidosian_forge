import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def _get_test_enforcer(self):
    test_rules = [policy.RuleDefault('foo', 'foo:bar=baz'), policy.RuleDefault('bar', 'bar:foo=baz')]
    enforcer = policy.Enforcer(self.conf)
    enforcer.register_defaults(test_rules)
    return enforcer