import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class RollingUpdatePolicyTest(common.HeatTestCase):

    def test_parse_without_update_policy(self):
        stack = utils.parse_stack(template)
        stack.validate()
        grp = stack['group1']
        self.assertFalse(grp.update_policy['rolling_update'])

    def test_parse_with_update_policy(self):
        tmpl = tmpl_with_updt_policy()
        stack = utils.parse_stack(tmpl)
        stack.validate()
        tmpl_grp = tmpl['resources']['group1']
        tmpl_policy = tmpl_grp['update_policy']['rolling_update']
        tmpl_batch_sz = int(tmpl_policy['max_batch_size'])
        grp = stack['group1']
        self.assertTrue(grp.update_policy)
        self.assertEqual(2, len(grp.update_policy))
        self.assertIn('rolling_update', grp.update_policy)
        policy = grp.update_policy['rolling_update']
        self.assertIsNotNone(policy)
        self.assertGreater(len(policy), 0)
        self.assertEqual(1, int(policy['min_in_service']))
        self.assertEqual(tmpl_batch_sz, int(policy['max_batch_size']))
        self.assertEqual(1, policy['pause_time'])

    def test_parse_with_default_update_policy(self):
        tmpl = tmpl_with_default_updt_policy()
        stack = utils.parse_stack(tmpl)
        stack.validate()
        grp = stack['group1']
        self.assertTrue(grp.update_policy)
        self.assertEqual(2, len(grp.update_policy))
        self.assertIn('rolling_update', grp.update_policy)
        policy = grp.update_policy['rolling_update']
        self.assertIsNotNone(policy)
        self.assertGreater(len(policy), 0)
        self.assertEqual(0, int(policy['min_in_service']))
        self.assertEqual(1, int(policy['max_batch_size']))
        self.assertEqual(0, policy['pause_time'])

    def test_parse_with_bad_update_policy(self):
        tmpl = tmpl_with_bad_updt_policy()
        stack = utils.parse_stack(tmpl)
        error = self.assertRaises(exception.StackValidationFailed, stack.validate)
        self.assertIn('foo', str(error))