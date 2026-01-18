import itertools
from heat.common import template_format
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class ReferencedAttrsTest(common.HeatTestCase):

    def setUp(self):
        super(ReferencedAttrsTest, self).setUp()
        parsed_tmpl = template_format.parse(tmpl6)
        self.stack = stack.Stack(utils.dummy_context(), 'test_stack', template.Template(parsed_tmpl))
        self.resA = self.stack['AResource']
        self.resB = self.stack['BResource']

    def test_referenced_attrs_resources(self):
        self.assertEqual(self.resA.referenced_attrs(in_resources=True, in_outputs=False), {('list', 1), ('nested_dict', 'dict', 'b')})
        self.assertEqual(self.resB.referenced_attrs(in_resources=True, in_outputs=False), set())

    def test_referenced_attrs_outputs(self):
        self.assertEqual(self.resA.referenced_attrs(in_resources=False, in_outputs=True), {('flat_dict', 'key2'), ('nested_dict', 'string')})
        self.assertEqual(self.resB.referenced_attrs(in_resources=False, in_outputs=True), {'attr_B3'})

    def test_referenced_attrs_single_output(self):
        self.assertEqual(self.resA.referenced_attrs(in_resources=False, in_outputs={'out1'}), {('flat_dict', 'key2'), ('nested_dict', 'string')})
        self.assertEqual(self.resB.referenced_attrs(in_resources=False, in_outputs={'out1'}), {'attr_B3'})
        self.assertEqual(self.resA.referenced_attrs(in_resources=False, in_outputs={'out2'}), set())
        self.assertEqual(self.resB.referenced_attrs(in_resources=False, in_outputs={'out2'}), set())

    def test_referenced_attrs_outputs_list(self):
        self.assertEqual(self.resA.referenced_attrs(in_resources=False, in_outputs={'out1', 'out2'}), {('flat_dict', 'key2'), ('nested_dict', 'string')})
        self.assertEqual(self.resB.referenced_attrs(in_resources=False, in_outputs={'out1', 'out2'}), {'attr_B3'})

    def test_referenced_attrs_both(self):
        self.assertEqual(self.resA.referenced_attrs(in_resources=True, in_outputs=True), {('list', 1), ('nested_dict', 'dict', 'b'), ('flat_dict', 'key2'), ('nested_dict', 'string')})
        self.assertEqual(self.resB.referenced_attrs(in_resources=True, in_outputs=True), {'attr_B3'})