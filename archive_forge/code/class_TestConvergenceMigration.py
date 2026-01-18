from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
class TestConvergenceMigration(common.HeatTestCase):

    def test_migration_to_convergence_engine(self):
        self.ctx = utils.dummy_context()
        self.stack = tools.get_stack('test_stack_convg', self.ctx, template=tools.string_template_five)
        self.stack.store()
        for r in self.stack.resources.values():
            r.store()
        self.stack.migrate_to_convergence()
        self.stack = self.stack.load(self.ctx, self.stack.id)
        self.assertTrue(self.stack.convergence)
        self.assertIsNone(self.stack.prev_raw_template_id)
        exp_required_by = {'A': ['C'], 'B': ['C'], 'C': ['D', 'E'], 'D': [], 'E': []}
        exp_requires = {'A': [], 'B': [], 'C': ['A', 'B'], 'D': ['C'], 'E': ['C']}
        exp_tmpl_id = self.stack.t.id

        def id_to_name(ids):
            names = []
            for r in self.stack.resources.values():
                if r.id in ids:
                    names.append(r.name)
            return names
        for r in self.stack.resources.values():
            self.assertEqual(sorted(exp_required_by[r.name]), sorted(r.required_by()))
            self.assertEqual(sorted(exp_requires[r.name]), sorted(id_to_name(r.requires)))
            self.assertEqual(exp_tmpl_id, r.current_template_id)