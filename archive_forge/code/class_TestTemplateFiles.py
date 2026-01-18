from heat.engine import template_files
from heat.tests import common
from heat.tests import utils
class TestTemplateFiles(common.HeatTestCase):

    def test_cache_miss(self):
        ctx = utils.dummy_context()
        tf1 = template_files.TemplateFiles(template_files_1)
        tf1.store(ctx)
        del tf1.files
        self.assertNotIn(tf1.files_id, template_files._d)
        self.assertEqual(template_files_1['template file 1'], tf1['template file 1'])
        self.assertEqual(template_files_1, template_files._d[tf1.files_id])

    def test_d_weakref_behaviour(self):
        ctx = utils.dummy_context()
        tf1 = template_files.TemplateFiles(template_files_1)
        tf1.store(ctx)
        tf2 = template_files.TemplateFiles(tf1)
        del tf1.files
        self.assertIn(tf2.files_id, template_files._d)
        del tf2.files
        self.assertNotIn(tf2.files_id, template_files._d)