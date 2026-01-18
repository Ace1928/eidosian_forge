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
class RulesTestCase(test_base.BaseTestCase):

    def test_init_basic(self):
        rules = policy.Rules()
        self.assertEqual({}, rules)
        self.assertIsNone(rules.default_rule)

    def test_init(self):
        rules = policy.Rules(dict(a=1, b=2, c=3), 'a')
        self.assertEqual(dict(a=1, b=2, c=3), rules)
        self.assertEqual('a', rules.default_rule)

    def test_no_default(self):
        rules = policy.Rules(dict(a=1, b=2, c=3))
        self.assertRaises(KeyError, lambda: rules['d'])

    def test_missing_default(self):
        rules = policy.Rules(dict(a=1, c=3), 'b')
        self.assertRaises(KeyError, lambda: rules['d'])

    def test_with_default(self):
        rules = policy.Rules(dict(a=1, b=2, c=3), 'b')
        self.assertEqual(2, rules['d'])

    def test_retrieval(self):
        rules = policy.Rules(dict(a=1, b=2, c=3), 'b')
        self.assertEqual(1, rules['a'])
        self.assertEqual(2, rules['b'])
        self.assertEqual(3, rules['c'])

    @mock.patch.object(_parser, 'parse_rule', lambda x: x)
    def test_load_json(self):
        exemplar = jsonutils.dumps({'admin_or_owner': [['role:admin'], ['project_id:%(project_id)s']], 'default': []})
        rules = policy.Rules.load(exemplar, 'default')
        self.assertEqual('default', rules.default_rule)
        self.assertEqual(dict(admin_or_owner=[['role:admin'], ['project_id:%(project_id)s']], default=[]), rules)

    @mock.patch.object(_parser, 'parse_rule', lambda x: x)
    def test_load_json_invalid_exc(self):
        exemplar = '{\n    "admin_or_owner": [["role:admin"], ["project_id:%(project_id)s"]],\n    "default": [\n}'
        self.assertRaises(ValueError, policy.Rules.load, exemplar, 'default')
        bad_but_acceptable = '{\n    \'admin_or_owner\': [["role:admin"], ["project_id:%(project_id)s"]],\n    \'default\': []\n}'
        self.assertTrue(policy.Rules.load(bad_but_acceptable, 'default'))
        bad_but_acceptable = '{\n    admin_or_owner: [["role:admin"], ["project_id:%(project_id)s"]],\n    default: []\n}'
        self.assertTrue(policy.Rules.load(bad_but_acceptable, 'default'))
        bad_but_acceptable = '{\n    admin_or_owner: [["role:admin"], ["project_id:%(project_id)s"]],\n    default: [],\n}'
        self.assertTrue(policy.Rules.load(bad_but_acceptable, 'default'))

    @mock.patch.object(_parser, 'parse_rule', lambda x: x)
    def test_load_empty_data(self):
        result = policy.Rules.load('', 'default')
        self.assertEqual(result, {})

    @mock.patch.object(_parser, 'parse_rule', lambda x: x)
    def test_load_yaml(self):
        exemplar = "\n# Define a custom rule.\nadmin_or_owner: role:admin or project_id:%(project_id)s\n# The default rule is used when there's no action defined.\ndefault: []\n"
        rules = policy.Rules.load(exemplar, 'default')
        self.assertEqual('default', rules.default_rule)
        self.assertEqual(dict(admin_or_owner='role:admin or project_id:%(project_id)s', default=[]), rules)

    @mock.patch.object(_parser, 'parse_rule', lambda x: x)
    def test_load_yaml_invalid_exc(self):
        exemplar = "{\n# Define a custom rule.\nadmin_or_owner: role:admin or project_id:%(project_id)s\n# The default rule is used when there's no action defined.\ndefault: [\n}"
        self.assertRaises(ValueError, policy.Rules.load, exemplar, 'default')

    @mock.patch.object(_parser, 'parse_rule', lambda x: x)
    def test_from_dict(self):
        expected = {'admin_or_owner': 'role:admin', 'default': '@'}
        rules = policy.Rules.from_dict(expected, 'default')
        self.assertEqual('default', rules.default_rule)
        self.assertEqual(expected, rules)

    def test_str(self):
        exemplar = jsonutils.dumps({'admin_or_owner': 'role:admin or project_id:%(project_id)s'}, indent=4)
        rules = policy.Rules(dict(admin_or_owner='role:admin or project_id:%(project_id)s'))
        self.assertEqual(exemplar, str(rules))

    def test_str_true(self):
        exemplar = jsonutils.dumps({'admin_or_owner': ''}, indent=4)
        rules = policy.Rules(dict(admin_or_owner=_checks.TrueCheck()))
        self.assertEqual(exemplar, str(rules))

    def test_load_json_deprecated(self):
        with self.assertWarnsRegex(DeprecationWarning, 'load_json\\(\\).*load\\(\\)'):
            policy.Rules.load_json(jsonutils.dumps({'default': ''}, 'default'))