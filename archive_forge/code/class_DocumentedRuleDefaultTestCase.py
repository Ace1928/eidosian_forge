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
class DocumentedRuleDefaultTestCase(base.PolicyBaseTestCase):

    def test_contain_operations(self):
        opt = policy.DocumentedRuleDefault(name='foo', check_str='rule:foo', description='foo_api', operations=[{'path': '/foo/', 'method': 'GET'}])
        self.assertEqual(1, len(opt.operations))

    def test_multiple_operations(self):
        opt = policy.DocumentedRuleDefault(name='foo', check_str='rule:foo', description='foo_api', operations=[{'path': '/foo/', 'method': 'GET'}, {'path': '/foo/', 'method': 'POST'}])
        self.assertEqual(2, len(opt.operations))

    def test_description_not_empty(self):
        invalid_desc = ''
        self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description=invalid_desc, operations=[{'path': '/foo/', 'method': 'GET'}])

    def test_operation_not_empty_list(self):
        invalid_op = []
        self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description='foo_api', operations=invalid_op)

    def test_operation_must_be_list(self):
        invalid_op = 'invalid_op'
        self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description='foo_api', operations=invalid_op)

    def test_operation_must_be_list_of_dicts(self):
        invalid_op = ['invalid_op']
        self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description='foo_api', operations=invalid_op)

    def test_operation_must_have_path(self):
        invalid_op = [{'method': 'POST'}]
        self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description='foo_api', operations=invalid_op)

    def test_operation_must_have_method(self):
        invalid_op = [{'path': '/foo/path/'}]
        self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description='foo_api', operations=invalid_op)

    def test_operation_must_contain_method_and_path_only(self):
        invalid_op = [{'path': '/some/path/', 'method': 'GET', 'break': 'me'}]
        self.assertRaises(policy.InvalidRuleDefault, policy.DocumentedRuleDefault, name='foo', check_str='rule:foo', description='foo_api', operations=invalid_op)