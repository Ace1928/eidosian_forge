import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class TestGetAttAllAttributes(common.HeatTestCase):
    scenarios = [('test_get_attr_all_attributes', dict(hot_tpl=hot_tpl_generic_resource_all_attrs, snippet={'Value': {'get_attr': ['resource1']}}, expected={'Value': {'Foo': 'resource1', 'foo': 'resource1'}}, raises=None)), ('test_get_attr_all_attributes_str', dict(hot_tpl=hot_tpl_generic_resource_all_attrs, snippet={'Value': {'get_attr': 'resource1'}}, expected='.Value.get_attr: Argument to "get_attr" must be a list', raises=exception.StackValidationFailed)), ('test_get_attr_all_attributes_invalid_resource_list', dict(hot_tpl=hot_tpl_generic_resource_all_attrs, snippet={'Value': {'get_attr': ['resource2']}}, raises=exception.InvalidTemplateReference, expected='The specified reference "resource2" (in unknown) is incorrect.')), ('test_get_attr_all_attributes_invalid_type', dict(hot_tpl=hot_tpl_generic_resource_all_attrs, snippet={'Value': {'get_attr': {'resource1': 'attr1'}}}, raises=exception.StackValidationFailed, expected='.Value.get_attr: Argument to "get_attr" must be a list')), ('test_get_attr_all_attributes_invalid_arg_str', dict(hot_tpl=hot_tpl_generic_resource_all_attrs, snippet={'Value': {'get_attr': ''}}, raises=exception.StackValidationFailed, expected='.Value.get_attr: Arguments to "get_attr" can be of the next forms: [resource_name] or [resource_name, attribute, (path), ...]')), ('test_get_attr_all_attributes_invalid_arg_list', dict(hot_tpl=hot_tpl_generic_resource_all_attrs, snippet={'Value': {'get_attr': []}}, raises=exception.StackValidationFailed, expected='.Value.get_attr: Arguments to "get_attr" can be of the next forms: [resource_name] or [resource_name, attribute, (path), ...]')), ('test_get_attr_all_attributes_standard', dict(hot_tpl=hot_tpl_generic_resource_all_attrs, snippet={'Value': {'get_attr': ['resource1', 'foo']}}, expected={'Value': 'resource1'}, raises=None)), ('test_get_attr_all_attrs_complex_attrs', dict(hot_tpl=hot_tpl_complex_attrs_all_attrs, snippet={'Value': {'get_attr': ['resource1']}}, expected={'Value': {'flat_dict': {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}, 'list': ['foo', 'bar'], 'nested_dict': {'dict': {'a': 1, 'b': 2, 'c': 3}, 'list': [1, 2, 3], 'string': 'abc'}, 'none': None}}, raises=None)), ('test_get_attr_all_attrs_complex_attrs_standard', dict(hot_tpl=hot_tpl_complex_attrs_all_attrs, snippet={'Value': {'get_attr': ['resource1', 'list', 1]}}, expected={'Value': 'bar'}, raises=None))]

    @staticmethod
    def resolve(snippet, template, stack):
        return function.resolve(template.parse(stack.defn, snippet))

    def test_get_attr_all_attributes(self):
        tmpl = template.Template(self.hot_tpl)
        stack = parser.Stack(utils.dummy_context(), 'test_get_attr', tmpl)
        stack.store()
        if self.raises is None:
            dep_attrs = list(function.dep_attrs(tmpl.parse(stack.defn, self.snippet), 'resource1'))
        else:
            dep_attrs = []
        stack.create()
        self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), stack.state)
        rsrc = stack['resource1']
        for action, status in ((rsrc.CREATE, rsrc.IN_PROGRESS), (rsrc.CREATE, rsrc.COMPLETE), (rsrc.RESUME, rsrc.IN_PROGRESS), (rsrc.RESUME, rsrc.COMPLETE), (rsrc.SUSPEND, rsrc.IN_PROGRESS), (rsrc.SUSPEND, rsrc.COMPLETE), (rsrc.UPDATE, rsrc.IN_PROGRESS), (rsrc.UPDATE, rsrc.COMPLETE), (rsrc.SNAPSHOT, rsrc.IN_PROGRESS), (rsrc.SNAPSHOT, rsrc.COMPLETE), (rsrc.CHECK, rsrc.IN_PROGRESS), (rsrc.CHECK, rsrc.COMPLETE), (rsrc.ADOPT, rsrc.IN_PROGRESS), (rsrc.ADOPT, rsrc.COMPLETE)):
            rsrc.state_set(action, status)
            with mock.patch.object(rsrc_defn.ResourceDefinition, 'dep_attrs') as mock_da:
                mock_da.return_value = dep_attrs
                node_data = rsrc.node_data()
            stk_defn.update_resource_data(stack.defn, rsrc.name, node_data)
            if self.raises is not None:
                ex = self.assertRaises(self.raises, self.resolve, self.snippet, tmpl, stack)
                self.assertEqual(self.expected, str(ex))
            else:
                self.assertEqual(self.expected, self.resolve(self.snippet, tmpl, stack))

    def test_stack_validate_outputs_get_all_attribute(self):
        hot_liberty_tpl = template_format.parse('\nheat_template_version: 2015-10-15\nresources:\n  resource1:\n    type: GenericResourceType\noutputs:\n  all_attr:\n    value: {get_attr: [resource1]}\n')
        stack = parser.Stack(utils.dummy_context(), 'test_outputs_get_all', template.Template(hot_liberty_tpl))
        stack.validate()