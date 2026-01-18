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
class StackParametersTest(common.HeatTestCase):
    """Test get_param function when stack was created from HOT template."""
    scenarios = [('Ref_string', dict(params={'foo': 'bar', 'blarg': 'wibble'}, snippet={'properties': {'prop1': {'Ref': 'foo'}, 'prop2': {'Ref': 'blarg'}}}, expected={'properties': {'prop1': 'bar', 'prop2': 'wibble'}})), ('get_param_string', dict(params={'foo': 'bar', 'blarg': 'wibble'}, snippet={'properties': {'prop1': {'get_param': 'foo'}, 'prop2': {'get_param': 'blarg'}}}, expected={'properties': {'prop1': 'bar', 'prop2': 'wibble'}})), ('get_list_attr', dict(params={'list': 'foo,bar'}, snippet={'properties': {'prop1': {'get_param': ['list', 1]}}}, expected={'properties': {'prop1': 'bar'}})), ('get_list_attr_string_index', dict(params={'list': 'foo,bar'}, snippet={'properties': {'prop1': {'get_param': ['list', '1']}}}, expected={'properties': {'prop1': 'bar'}})), ('get_flat_dict_attr', dict(params={'flat_dict': {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}}, snippet={'properties': {'prop1': {'get_param': ['flat_dict', 'key2']}}}, expected={'properties': {'prop1': 'val2'}})), ('get_nested_attr_list', dict(params={'nested_dict': {'list': [1, 2, 3], 'string': 'abc', 'dict': {'a': 1, 'b': 2, 'c': 3}}}, snippet={'properties': {'prop1': {'get_param': ['nested_dict', 'list', 0]}}}, expected={'properties': {'prop1': 1}})), ('get_nested_attr_dict', dict(params={'nested_dict': {'list': [1, 2, 3], 'string': 'abc', 'dict': {'a': 1, 'b': 2, 'c': 3}}}, snippet={'properties': {'prop1': {'get_param': ['nested_dict', 'dict', 'a']}}}, expected={'properties': {'prop1': 1}})), ('get_attr_none', dict(params={'none': None}, snippet={'properties': {'prop1': {'get_param': ['none', 'who_cares']}}}, expected={'properties': {'prop1': ''}})), ('pseudo_stack_id', dict(params={}, snippet={'properties': {'prop1': {'get_param': 'OS::stack_id'}}}, expected={'properties': {'prop1': '1ba8c334-2297-4312-8c7c-43763a988ced'}})), ('pseudo_stack_name', dict(params={}, snippet={'properties': {'prop1': {'get_param': 'OS::stack_name'}}}, expected={'properties': {'prop1': 'test'}})), ('pseudo_project_id', dict(params={}, snippet={'properties': {'prop1': {'get_param': 'OS::project_id'}}}, expected={'properties': {'prop1': '9913ef0a-b8be-4b33-b574-9061441bd373'}}))]
    props_template = template_format.parse("\n    heat_template_version: 2013-05-23\n    parameters:\n        foo:\n            type: string\n            default: ''\n        blarg:\n            type: string\n            default: ''\n        list:\n            type: comma_delimited_list\n            default: ''\n        flat_dict:\n            type: json\n            default: {}\n        nested_dict:\n            type: json\n            default: {}\n        none:\n            type: string\n            default: 'default'\n    ")

    def test_param_refs(self):
        """Test if parameter references work."""
        env = environment.Environment(self.params)
        tmpl = template.Template(self.props_template, env=env)
        stack = parser.Stack(utils.dummy_context(), 'test', tmpl, stack_id='1ba8c334-2297-4312-8c7c-43763a988ced', tenant_id='9913ef0a-b8be-4b33-b574-9061441bd373')
        self.assertEqual(self.expected, function.resolve(tmpl.parse(stack.defn, self.snippet)))