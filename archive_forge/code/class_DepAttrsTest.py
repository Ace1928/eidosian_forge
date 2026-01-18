import itertools
from heat.common import template_format
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class DepAttrsTest(common.HeatTestCase):
    scenarios = [('no_attr', dict(tmpl=tmpl1, expected={'AResource': set()})), ('one_res_one_attr', dict(tmpl=tmpl2, expected={'AResource': {'attr_A1'}, 'BResource': set()})), ('one_res_several_attrs', dict(tmpl=tmpl3, expected={'AResource': {'attr_A1', 'attr_A2', 'attr_A3', 'meta_A1', 'meta_A2'}, 'BResource': set()})), ('several_res_one_attr', dict(tmpl=tmpl4, expected={'AResource': {'attr_A1'}, 'BResource': {'attr_B1'}, 'CResource': {'attr_C1'}, 'DResource': set()})), ('several_res_several_attrs', dict(tmpl=tmpl5, expected={'AResource': {'attr_A1', 'attr_A2', 'meta_A1'}, 'BResource': {'attr_B1', 'attr_B2', 'meta_B2'}, 'CResource': set()})), ('nested_attr', dict(tmpl=tmpl6, expected={'AResource': set([(u'list', 1), (u'nested_dict', u'dict', u'b')]), 'BResource': set([])})), ('several_res_several_attrs_and_all_attrs', dict(tmpl=tmpl7, expected={'AResource': {'attr_A1', 'attr_A2', 'meta_A1'}, 'BResource': {'attr_B1', 'attr_B2', 'meta_B2'}, 'CResource': set()}))]

    def setUp(self):
        super(DepAttrsTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.parsed_tmpl = template_format.parse(self.tmpl)
        self.stack = stack.Stack(self.ctx, 'test_stack', template.Template(self.parsed_tmpl))

    def test_dep_attrs(self):
        for res in self.stack.values():
            definitions = (self.stack.defn.resource_definition(n) for n in self.parsed_tmpl['resources'])
            self.assertEqual(self.expected[res.name], set(itertools.chain.from_iterable((d.dep_attrs(res.name) for d in definitions))))

    def test_all_dep_attrs(self):
        for res in self.stack.values():
            definitions = (self.stack.defn.resource_definition(n) for n in self.parsed_tmpl['resources'])
            attrs = set(itertools.chain.from_iterable((d.dep_attrs(res.name, load_all=True) for d in definitions)))
            self.assertEqual(self.expected[res.name], attrs)