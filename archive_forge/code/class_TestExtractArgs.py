import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class TestExtractArgs(common.HeatTestCase):

    def test_timeout_extract(self):
        p = {'timeout_mins': '5'}
        args = api.extract_args(p)
        self.assertEqual(5, args['timeout_mins'])

    def test_timeout_extract_zero(self):
        p = {'timeout_mins': '0'}
        args = api.extract_args(p)
        self.assertNotIn('timeout_mins', args)

    def test_timeout_extract_garbage(self):
        p = {'timeout_mins': 'wibble'}
        args = api.extract_args(p)
        self.assertNotIn('timeout_mins', args)

    def test_timeout_extract_none(self):
        p = {'timeout_mins': None}
        args = api.extract_args(p)
        self.assertNotIn('timeout_mins', args)

    def test_timeout_extract_negative(self):
        p = {'timeout_mins': '-100'}
        error = self.assertRaises(ValueError, api.extract_args, p)
        self.assertIn('Invalid timeout value', str(error))

    def test_timeout_extract_not_present(self):
        args = api.extract_args({})
        self.assertNotIn('timeout_mins', args)

    def test_adopt_stack_data_extract_present(self):
        p = {'adopt_stack_data': json.dumps({'Resources': {}})}
        args = api.extract_args(p)
        self.assertTrue(args.get('adopt_stack_data'))

    def test_invalid_adopt_stack_data(self):
        params = {'adopt_stack_data': json.dumps('foo')}
        exc = self.assertRaises(ValueError, api.extract_args, params)
        self.assertIn('Invalid adopt data', str(exc))

    def test_adopt_stack_data_extract_not_present(self):
        args = api.extract_args({})
        self.assertNotIn('adopt_stack_data', args)

    def test_disable_rollback_extract_true(self):
        args = api.extract_args({'disable_rollback': True})
        self.assertIn('disable_rollback', args)
        self.assertTrue(args.get('disable_rollback'))
        args = api.extract_args({'disable_rollback': 'True'})
        self.assertIn('disable_rollback', args)
        self.assertTrue(args.get('disable_rollback'))
        args = api.extract_args({'disable_rollback': 'true'})
        self.assertIn('disable_rollback', args)
        self.assertTrue(args.get('disable_rollback'))

    def test_disable_rollback_extract_false(self):
        args = api.extract_args({'disable_rollback': False})
        self.assertIn('disable_rollback', args)
        self.assertFalse(args.get('disable_rollback'))
        args = api.extract_args({'disable_rollback': 'False'})
        self.assertIn('disable_rollback', args)
        self.assertFalse(args.get('disable_rollback'))
        args = api.extract_args({'disable_rollback': 'false'})
        self.assertIn('disable_rollback', args)
        self.assertFalse(args.get('disable_rollback'))

    def test_disable_rollback_extract_bad(self):
        self.assertRaises(ValueError, api.extract_args, {'disable_rollback': 'bad'})

    def test_tags_extract(self):
        p = {'tags': ['tag1', 'tag2']}
        args = api.extract_args(p)
        self.assertEqual(['tag1', 'tag2'], args['tags'])

    def test_tags_extract_not_present(self):
        args = api.extract_args({})
        self.assertNotIn('tags', args)

    def test_tags_extract_not_map(self):
        p = {'tags': {'foo': 'bar'}}
        exc = self.assertRaises(ValueError, api.extract_args, p)
        self.assertIn('Invalid tags, not a list: ', str(exc))

    def test_tags_extract_not_string(self):
        p = {'tags': ['tag1', 2]}
        exc = self.assertRaises(ValueError, api.extract_args, p)
        self.assertIn('Invalid tag, "2" is not a string', str(exc))

    def test_tags_extract_over_limit(self):
        p = {'tags': ['tag1', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa']}
        exc = self.assertRaises(ValueError, api.extract_args, p)
        self.assertIn('Invalid tag, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" is longer than 80 characters', str(exc))

    def test_tags_extract_comma(self):
        p = {'tags': ['tag1', 'tag2,']}
        exc = self.assertRaises(ValueError, api.extract_args, p)
        self.assertIn('Invalid tag, "tag2," contains a comma', str(exc))