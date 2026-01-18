import copy
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions
from heat.engine import environment
from heat.engine import function
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class ValidateGetAttTest(common.HeatTestCase):

    def setUp(self):
        super(ValidateGetAttTest, self).setUp()
        env = environment.Environment()
        env.load({u'resource_registry': {u'OS::Test::GenericResource': u'GenericResourceType'}})
        env.load({u'resource_registry': {u'OS::Test::FakeResource': u'OverwrittenFnGetAttType'}})
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'test_rsrc': {'Type': 'OS::Test::GenericResource'}, 'get_att_rsrc': {'Type': 'OS::Heat::Value', 'Properties': {'value': {'Fn::GetAtt': ['test_rsrc', 'Foo']}}}}}, env=env)
        self.stack = stack.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()))
        self.rsrc = self.stack['test_rsrc']
        self.stack.validate()

    def test_resource_is_appear_in_stack(self):
        func = functions.GetAtt(self.stack.defn, 'Fn::GetAtt', [self.rsrc.name, 'Foo'])
        self.assertIsNone(func.validate())

    def test_resource_is_not_appear_in_stack(self):
        self.stack.remove_resource(self.rsrc.name)
        func = functions.GetAtt(self.stack.defn, 'Fn::GetAtt', [self.rsrc.name, 'Foo'])
        ex = self.assertRaises(exception.InvalidTemplateReference, func.validate)
        self.assertEqual('The specified reference "test_rsrc" (in unknown) is incorrect.', str(ex))

    def test_resource_no_attribute_with_default_fn_get_att(self):
        res_defn = rsrc_defn.ResourceDefinition('test_rsrc', 'ResWithStringPropAndAttr')
        self.rsrc = resource.Resource('test_rsrc', res_defn, self.stack)
        self.stack.add_resource(self.rsrc)
        stk_defn.update_resource_data(self.stack.defn, self.rsrc.name, self.rsrc.node_data())
        self.stack.validate()
        func = functions.GetAtt(self.stack.defn, 'Fn::GetAtt', [self.rsrc.name, 'Bar'])
        ex = self.assertRaises(exception.InvalidTemplateAttribute, func.validate)
        self.assertEqual('The Referenced Attribute (test_rsrc Bar) is incorrect.', str(ex))

    def test_resource_no_attribute_with_overwritten_fn_get_att(self):
        res_defn = rsrc_defn.ResourceDefinition('test_rsrc', 'OS::Test::FakeResource')
        self.rsrc = resource.Resource('test_rsrc', res_defn, self.stack)
        self.rsrc.attributes_schema = {}
        self.stack.add_resource(self.rsrc)
        stk_defn.update_resource_data(self.stack.defn, self.rsrc.name, self.rsrc.node_data())
        self.stack.validate()
        func = functions.GetAtt(self.stack.defn, 'Fn::GetAtt', [self.rsrc.name, 'Foo'])
        self.assertIsNone(func.validate())

    def test_get_attr_without_attribute_name(self):
        ex = self.assertRaises(ValueError, functions.GetAtt, self.stack.defn, 'Fn::GetAtt', [self.rsrc.name])
        self.assertEqual('Arguments to "Fn::GetAtt" must be of the form [resource_name, attribute]', str(ex))