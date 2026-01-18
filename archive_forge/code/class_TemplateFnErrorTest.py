import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class TemplateFnErrorTest(common.HeatTestCase):
    scenarios = [('select_from_list_not_int', dict(expect=TypeError, snippet={'Fn::Select': ['one', ['foo', 'bar']]})), ('select_from_dict_not_str', dict(expect=TypeError, snippet={'Fn::Select': [1, {'red': 'robin', 're': 'foo'}]})), ('select_from_serialized_json_wrong', dict(expect=ValueError, snippet={'Fn::Select': ['not', 'no json']})), ('select_wrong_num_args_1', dict(expect=exception.StackValidationFailed, snippet={'Fn::Select': []})), ('select_wrong_num_args_2', dict(expect=exception.StackValidationFailed, snippet={'Fn::Select': ['4']})), ('select_wrong_num_args_3', dict(expect=exception.StackValidationFailed, snippet={'Fn::Select': ['foo', {'foo': 'bar'}, '']})), ('select_wrong_num_args_4', dict(expect=TypeError, snippet={'Fn::Select': [['f'], {'f': 'food'}]})), ('split_no_delim', dict(expect=exception.StackValidationFailed, snippet={'Fn::Split': ['foo, bar, achoo']})), ('split_no_list', dict(expect=exception.StackValidationFailed, snippet={'Fn::Split': 'foo, bar, achoo'})), ('base64_list', dict(expect=TypeError, snippet={'Fn::Base64': ['foobar']})), ('base64_dict', dict(expect=TypeError, snippet={'Fn::Base64': {'foo': 'bar'}})), ('replace_list_value', dict(expect=TypeError, snippet={'Fn::Replace': [{'$var1': 'foo', '%var2%': ['bar']}, '$var1 is %var2%']})), ('replace_list_mapping', dict(expect=exception.StackValidationFailed, snippet={'Fn::Replace': [['var1', 'foo', 'var2', 'bar'], '$var1 is ${var2}']})), ('replace_dict', dict(expect=exception.StackValidationFailed, snippet={'Fn::Replace': {}})), ('replace_missing_template', dict(expect=exception.StackValidationFailed, snippet={'Fn::Replace': [['var1', 'foo', 'var2', 'bar']]})), ('replace_none_template', dict(expect=exception.StackValidationFailed, snippet={'Fn::Replace': [['var2', 'bar'], None]})), ('replace_list_string', dict(expect=TypeError, snippet={'Fn::Replace': [{'var1': 'foo', 'var2': 'bar'}, ['$var1 is ${var2}']]})), ('join_string', dict(expect=TypeError, snippet={'Fn::Join': [' ', 'foo']})), ('join_dict', dict(expect=TypeError, snippet={'Fn::Join': [' ', {'foo': 'bar'}]})), ('join_wrong_num_args_1', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': []})), ('join_wrong_num_args_2', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': [' ']})), ('join_wrong_num_args_3', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': [' ', {'foo': 'bar'}, '']})), ('join_string_nodelim', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': 'o'})), ('join_string_nodelim_1', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': 'oh'})), ('join_string_nodelim_2', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': 'ohh'})), ('join_dict_nodelim1', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': {'foo': 'bar'}})), ('join_dict_nodelim2', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': {'foo': 'bar', 'blarg': 'wibble'}})), ('join_dict_nodelim3', dict(expect=exception.StackValidationFailed, snippet={'Fn::Join': {'foo': 'bar', 'blarg': 'wibble', 'baz': 'quux'}})), ('member_list2map_no_key_or_val', dict(expect=exception.StackValidationFailed, snippet={'Fn::MemberListToMap': ['Key', ['.member.2.Key=metric', '.member.2.Value=cpu', '.member.5.Key=size', '.member.5.Value=56']]})), ('member_list2map_no_list', dict(expect=exception.StackValidationFailed, snippet={'Fn::MemberListToMap': ['Key', '.member.2.Key=metric']})), ('member_list2map_not_string', dict(expect=exception.StackValidationFailed, snippet={'Fn::MemberListToMap': ['Name', ['Value'], ['.member.0.Name=metric', '.member.0.Value=cpu', '.member.1.Name=size', '.member.1.Value=56']]}))]

    def test_bad_input(self):
        tmpl = template.Template(empty_template)

        def resolve(s):
            return TemplateTest.resolve(s, tmpl)
        error = self.assertRaises(self.expect, resolve, self.snippet)
        self.assertIn(next(iter(self.snippet)), str(error))