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
class TemplateTest(common.HeatTestCase):

    def setUp(self):
        super(TemplateTest, self).setUp()
        self.ctx = utils.dummy_context()

    @staticmethod
    def resolve(snippet, template, stack=None):
        return function.resolve(template.parse(stack and stack.defn, snippet))

    @staticmethod
    def resolve_condition(snippet, template, stack=None):
        return function.resolve(template.parse_condition(stack and stack.defn, snippet))

    def test_defaults(self):
        empty = template.Template(empty_template)
        self.assertNotIn('AWSTemplateFormatVersion', empty)
        self.assertEqual('No description', empty['Description'])
        self.assertEqual({}, empty['Mappings'])
        self.assertEqual({}, empty['Resources'])
        self.assertEqual({}, empty['Outputs'])

    def test_aws_version(self):
        tmpl = template.Template(mapping_template)
        self.assertEqual(('AWSTemplateFormatVersion', '2010-09-09'), tmpl.version)

    def test_heat_version(self):
        tmpl = template.Template(resource_template)
        self.assertEqual(('HeatTemplateFormatVersion', '2012-12-12'), tmpl.version)

    def test_invalid_hot_version(self):
        invalid_hot_version_tmp = template_format.parse('{\n            "heat_template_version" : "2012-12-12",\n            }')
        init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_hot_version_tmp)
        valid_versions = ['2013-05-23', '2014-10-16', '2015-04-30', '2015-10-15', '2016-04-08', '2016-10-14', '2017-02-24', '2017-09-01', '2018-03-02', '2018-08-31', '2021-04-16', 'newton', 'ocata', 'pike', 'queens', 'rocky', 'wallaby']
        ex_error_msg = 'The template version is invalid: "heat_template_version: 2012-12-12". "heat_template_version" should be one of: %s' % ', '.join(valid_versions)
        self.assertEqual(ex_error_msg, str(init_ex))

    def test_invalid_version_not_in_hot_versions(self):
        invalid_hot_version_tmp = template_format.parse('{\n            "heat_template_version" : "2012-12-12",\n            }')
        versions = {('heat_template_version', '2013-05-23'): hot_t.HOTemplate20130523, ('heat_template_version', '2013-06-23'): hot_t.HOTemplate20130523}
        temp_copy = copy.deepcopy(template._template_classes)
        template._template_classes = versions
        init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_hot_version_tmp)
        ex_error_msg = 'The template version is invalid: "heat_template_version: 2012-12-12". "heat_template_version" should be one of: 2013-05-23, 2013-06-23'
        self.assertEqual(ex_error_msg, str(init_ex))
        template._template_classes = temp_copy

    def test_invalid_aws_version(self):
        invalid_aws_version_tmp = template_format.parse('{\n            "AWSTemplateFormatVersion" : "2012-12-12",\n            }')
        init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_aws_version_tmp)
        ex_error_msg = 'The template version is invalid: "AWSTemplateFormatVersion: 2012-12-12". "AWSTemplateFormatVersion" should be: 2010-09-09'
        self.assertEqual(ex_error_msg, str(init_ex))

    def test_invalid_version_not_in_aws_versions(self):
        invalid_aws_version_tmp = template_format.parse('{\n            "AWSTemplateFormatVersion" : "2012-12-12",\n            }')
        versions = {('AWSTemplateFormatVersion', '2010-09-09'): cfn_t.CfnTemplate, ('AWSTemplateFormatVersion', '2011-06-23'): cfn_t.CfnTemplate}
        temp_copy = copy.deepcopy(template._template_classes)
        template._template_classes = versions
        init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_aws_version_tmp)
        ex_error_msg = 'The template version is invalid: "AWSTemplateFormatVersion: 2012-12-12". "AWSTemplateFormatVersion" should be one of: 2010-09-09, 2011-06-23'
        self.assertEqual(ex_error_msg, str(init_ex))
        template._template_classes = temp_copy

    def test_invalid_heat_version(self):
        invalid_heat_version_tmp = template_format.parse('{\n            "HeatTemplateFormatVersion" : "2010-09-09",\n            }')
        init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_heat_version_tmp)
        ex_error_msg = 'The template version is invalid: "HeatTemplateFormatVersion: 2010-09-09". "HeatTemplateFormatVersion" should be: 2012-12-12'
        self.assertEqual(ex_error_msg, str(init_ex))

    def test_invalid_version_not_in_heat_versions(self):
        invalid_heat_version_tmp = template_format.parse('{\n            "HeatTemplateFormatVersion" : "2010-09-09",\n            }')
        versions = {('HeatTemplateFormatVersion', '2012-12-12'): cfn_t.CfnTemplate, ('HeatTemplateFormatVersion', '2014-12-12'): cfn_t.CfnTemplate}
        temp_copy = copy.deepcopy(template._template_classes)
        template._template_classes = versions
        init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_heat_version_tmp)
        ex_error_msg = 'The template version is invalid: "HeatTemplateFormatVersion: 2010-09-09". "HeatTemplateFormatVersion" should be one of: 2012-12-12, 2014-12-12'
        self.assertEqual(ex_error_msg, str(init_ex))
        template._template_classes = temp_copy

    def test_invalid_template(self):
        scanner_error = '\n            1\n            Mappings:\n              ValidMapping:\n                TestKey: TestValue\n            '
        parser_error = '\n            Mappings:\n              ValidMapping:\n                TestKey: {TestKey1: "Value1" TestKey2: "Value2"}\n            '
        self.assertRaises(ValueError, template_format.parse, scanner_error)
        self.assertRaises(ValueError, template_format.parse, parser_error)

    def test_invalid_section(self):
        tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Foo': ['Bar']})
        self.assertNotIn('Foo', tmpl)

    def test_find_in_map(self):
        tmpl = template.Template(mapping_template)
        stk = stack.Stack(self.ctx, 'test', tmpl)
        find = {'Fn::FindInMap': ['ValidMapping', 'TestKey', 'TestValue']}
        self.assertEqual('wibble', self.resolve(find, tmpl, stk))

    def test_find_in_invalid_map(self):
        tmpl = template.Template(mapping_template)
        stk = stack.Stack(self.ctx, 'test', tmpl)
        finds = ({'Fn::FindInMap': ['InvalidMapping', 'ValueList', 'foo']}, {'Fn::FindInMap': ['InvalidMapping', 'ValueString', 'baz']}, {'Fn::FindInMap': ['MapList', 'foo', 'bar']}, {'Fn::FindInMap': ['MapString', 'foo', 'bar']})
        for find in finds:
            self.assertRaises((KeyError, TypeError), self.resolve, find, tmpl, stk)

    def test_bad_find_in_map(self):
        tmpl = template.Template(mapping_template)
        stk = stack.Stack(self.ctx, 'test', tmpl)
        finds = ({'Fn::FindInMap': 'String'}, {'Fn::FindInMap': {'Dict': 'String'}}, {'Fn::FindInMap': ['ShortList', 'foo']}, {'Fn::FindInMap': ['ReallyShortList']})
        for find in finds:
            self.assertRaises(exception.StackValidationFailed, self.resolve, find, tmpl, stk)

    def test_param_refs(self):
        env = environment.Environment({'foo': 'bar', 'blarg': 'wibble'})
        tmpl = template.Template(parameter_template, env=env)
        stk = stack.Stack(self.ctx, 'test', tmpl)
        p_snippet = {'Ref': 'foo'}
        self.assertEqual('bar', self.resolve(p_snippet, tmpl, stk))

    def test_param_ref_missing(self):
        env = environment.Environment({'foo': 'bar'})
        tmpl = template.Template(parameter_template, env=env)
        stk = stack.Stack(self.ctx, 'test', tmpl)
        tmpl.env = environment.Environment({})
        stk.defn.parameters = cfn_p.CfnParameters(stk.identifier(), tmpl)
        snippet = {'Ref': 'foo'}
        self.assertRaises(exception.UserParameterMissing, self.resolve, snippet, tmpl, stk)

    def test_resource_refs(self):
        tmpl = template.Template(resource_template)
        stk = stack.Stack(self.ctx, 'test', tmpl)
        stk.validate()
        data = node_data.NodeData.from_dict({'reference_id': 'bar'})
        stk_defn.update_resource_data(stk.defn, 'foo', data)
        r_snippet = {'Ref': 'foo'}
        self.assertEqual('bar', self.resolve(r_snippet, tmpl, stk))

    def test_resource_refs_param(self):
        tmpl = template.Template(resource_template)
        stk = stack.Stack(self.ctx, 'test', tmpl)
        p_snippet = {'Ref': 'baz'}
        parsed = tmpl.parse(stk.defn, p_snippet)
        self.assertIsInstance(parsed, cfn_funcs.ParamRef)

    def test_select_from_list(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': ['1', ['foo', 'bar']]}
        self.assertEqual('bar', self.resolve(data, tmpl))

    def test_select_from_list_integer_index(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': [1, ['foo', 'bar']]}
        self.assertEqual('bar', self.resolve(data, tmpl))

    def test_select_from_list_out_of_bound(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': ['0', ['foo', 'bar']]}
        self.assertEqual('foo', self.resolve(data, tmpl))
        data = {'Fn::Select': ['1', ['foo', 'bar']]}
        self.assertEqual('bar', self.resolve(data, tmpl))
        data = {'Fn::Select': ['2', ['foo', 'bar']]}
        self.assertEqual('', self.resolve(data, tmpl))

    def test_select_from_dict(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': ['red', {'red': 'robin', 're': 'foo'}]}
        self.assertEqual('robin', self.resolve(data, tmpl))

    def test_select_int_from_dict(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': ['2', {'1': 'bar', '2': 'foo'}]}
        self.assertEqual('foo', self.resolve(data, tmpl))

    def test_select_from_none(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': ['red', None]}
        self.assertEqual('', self.resolve(data, tmpl))

    def test_select_from_dict_not_existing(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': ['green', {'red': 'robin', 're': 'foo'}]}
        self.assertEqual('', self.resolve(data, tmpl))

    def test_select_from_serialized_json_map(self):
        tmpl = template.Template(empty_template)
        js = json.dumps({'red': 'robin', 're': 'foo'})
        data = {'Fn::Select': ['re', js]}
        self.assertEqual('foo', self.resolve(data, tmpl))

    def test_select_from_serialized_json_list(self):
        tmpl = template.Template(empty_template)
        js = json.dumps(['foo', 'fee', 'fum'])
        data = {'Fn::Select': ['0', js]}
        self.assertEqual('foo', self.resolve(data, tmpl))

    def test_select_empty_string(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Select': ['0', '']}
        self.assertEqual('', self.resolve(data, tmpl))
        data = {'Fn::Select': ['1', '']}
        self.assertEqual('', self.resolve(data, tmpl))
        data = {'Fn::Select': ['one', '']}
        self.assertEqual('', self.resolve(data, tmpl))

    def test_equals(self):
        tpl = template_format.parse("\n        AWSTemplateFormatVersion: 2010-09-09\n        Parameters:\n          env_type:\n            Type: String\n            Default: 'test'\n        ")
        snippet = {'Fn::Equals': [{'Ref': 'env_type'}, 'prod']}
        tmpl = template.Template(tpl)
        stk = stack.Stack(utils.dummy_context(), 'test_equals_false', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertFalse(resolved)
        tmpl = template.Template(tpl, env=environment.Environment({'env_type': 'prod'}))
        stk = stack.Stack(utils.dummy_context(), 'test_equals_true', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertTrue(resolved)

    def test_equals_invalid_args(self):
        tmpl = template.Template(aws_empty_template)
        snippet = {'Fn::Equals': ['test', 'prod', 'invalid']}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        error_msg = '.Fn::Equals: Arguments to "Fn::Equals" must be of the form: [value_1, value_2]'
        self.assertIn(error_msg, str(exc))
        snippet = {'Fn::Equals': {'equal': False}}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        self.assertIn(error_msg, str(exc))

    def test_not(self):
        tpl = template_format.parse("\n        AWSTemplateFormatVersion: 2010-09-09\n        Parameters:\n          env_type:\n            Type: String\n            Default: 'test'\n        ")
        snippet = {'Fn::Not': [{'Fn::Equals': [{'Ref': 'env_type'}, 'prod']}]}
        tmpl = template.Template(tpl)
        stk = stack.Stack(utils.dummy_context(), 'test_not_true', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertTrue(resolved)
        tmpl = template.Template(tpl, env=environment.Environment({'env_type': 'prod'}))
        stk = stack.Stack(utils.dummy_context(), 'test_not_false', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertFalse(resolved)

    def test_not_invalid_args(self):
        tmpl = template.Template(aws_empty_template)
        stk = stack.Stack(utils.dummy_context(), 'test_not_invalid', tmpl)
        snippet = {'Fn::Not': ['invalid_arg']}
        exc = self.assertRaises(ValueError, self.resolve_condition, snippet, tmpl, stk)
        error_msg = 'Invalid condition "invalid_arg"'
        self.assertIn(error_msg, str(exc))
        snippet = {'Fn::Not': 'invalid'}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        error_msg = 'Arguments to "Fn::Not" must be '
        self.assertIn(error_msg, str(exc))
        snippet = {'Fn::Not': ['cd1', 'cd2']}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        error_msg = 'Arguments to "Fn::Not" must be '
        self.assertIn(error_msg, str(exc))

    def test_and(self):
        tpl = template_format.parse("\n        AWSTemplateFormatVersion: 2010-09-09\n        Parameters:\n          env_type:\n            Type: String\n            Default: 'test'\n          zone:\n            Type: String\n            Default: 'shanghai'\n        ")
        snippet = {'Fn::And': [{'Fn::Equals': [{'Ref': 'env_type'}, 'prod']}, {'Fn::Not': [{'Fn::Equals': [{'Ref': 'zone'}, 'beijing']}]}]}
        tmpl = template.Template(tpl)
        stk = stack.Stack(utils.dummy_context(), 'test_and_false', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertFalse(resolved)
        tmpl = template.Template(tpl, env=environment.Environment({'env_type': 'prod'}))
        stk = stack.Stack(utils.dummy_context(), 'test_and_true', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertTrue(resolved)
        tmpl = template.Template(tpl, env=environment.Environment({'env_type': 'prod', 'zone': 'beijing'}))
        stk = stack.Stack(utils.dummy_context(), 'test_and_false', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertFalse(resolved)

    def test_and_invalid_args(self):
        tmpl = template.Template(aws_empty_template)
        error_msg = 'The minimum number of condition arguments to "Fn::And" is 2.'
        snippet = {'Fn::And': ['invalid_arg']}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        self.assertIn(error_msg, str(exc))
        error_msg = 'Arguments to "Fn::And" must be'
        snippet = {'Fn::And': 'invalid'}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        self.assertIn(error_msg, str(exc))
        stk = stack.Stack(utils.dummy_context(), 'test_and_invalid', tmpl)
        snippet = {'Fn::And': ['cd1', True]}
        exc = self.assertRaises(ValueError, self.resolve_condition, snippet, tmpl, stk)
        error_msg = 'Invalid condition "cd1"'
        self.assertIn(error_msg, str(exc))

    def test_or(self):
        tpl = template_format.parse("\n        AWSTemplateFormatVersion: 2010-09-09\n        Parameters:\n          zone:\n            Type: String\n            Default: 'guangzhou'\n        ")
        snippet = {'Fn::Or': [{'Fn::Equals': [{'Ref': 'zone'}, 'shanghai']}, {'Fn::Equals': [{'Ref': 'zone'}, 'beijing']}]}
        tmpl = template.Template(tpl)
        stk = stack.Stack(utils.dummy_context(), 'test_or_false', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertFalse(resolved)
        tmpl = template.Template(tpl, env=environment.Environment({'zone': 'beijing'}))
        stk = stack.Stack(utils.dummy_context(), 'test_or_true', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertTrue(resolved)
        tmpl = template.Template(tpl, env=environment.Environment({'zone': 'shanghai'}))
        stk = stack.Stack(utils.dummy_context(), 'test_or_true', tmpl)
        resolved = self.resolve_condition(snippet, tmpl, stk)
        self.assertTrue(resolved)

    def test_or_invalid_args(self):
        tmpl = template.Template(aws_empty_template)
        error_msg = 'The minimum number of condition arguments to "Fn::Or" is 2.'
        snippet = {'Fn::Or': ['invalid_arg']}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        self.assertIn(error_msg, str(exc))
        error_msg = 'Arguments to "Fn::Or" must be'
        snippet = {'Fn::Or': 'invalid'}
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve_condition, snippet, tmpl)
        self.assertIn(error_msg, str(exc))
        stk = stack.Stack(utils.dummy_context(), 'test_or_invalid', tmpl)
        snippet = {'Fn::Or': ['invalid_cd', True]}
        exc = self.assertRaises(ValueError, self.resolve_condition, snippet, tmpl, stk)
        error_msg = 'Invalid condition "invalid_cd"'
        self.assertIn(error_msg, str(exc))

    def test_join(self):
        tmpl = template.Template(empty_template)
        join = {'Fn::Join': [' ', ['foo', 'bar']]}
        self.assertEqual('foo bar', self.resolve(join, tmpl))

    def test_split_ok(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Split': [';', 'foo; bar; achoo']}
        self.assertEqual(['foo', ' bar', ' achoo'], self.resolve(data, tmpl))

    def test_split_no_delim_in_str(self):
        tmpl = template.Template(empty_template)
        data = {'Fn::Split': [';', 'foo, bar, achoo']}
        self.assertEqual(['foo, bar, achoo'], self.resolve(data, tmpl))

    def test_base64(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::Base64': 'foobar'}
        self.assertEqual('foobar', self.resolve(snippet, tmpl))

    def test_get_azs(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::GetAZs': ''}
        self.assertEqual(['nova'], self.resolve(snippet, tmpl))

    def test_get_azs_with_stack(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::GetAZs': ''}
        stk = stack.Stack(self.ctx, 'test_stack', template.Template(empty_template))
        fc = fakes_nova.FakeClient()
        self.patchobject(nova.NovaClientPlugin, 'client', return_value=fc)
        self.assertEqual(['nova1'], self.resolve(snippet, tmpl, stk))

    def test_replace_string_values(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::Replace': [{'$var1': 'foo', '%var2%': 'bar'}, '$var1 is %var2%']}
        self.assertEqual('foo is bar', self.resolve(snippet, tmpl))

    def test_replace_number_values(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::Replace': [{'$var1': 1, '%var2%': 2}, '$var1 is not %var2%']}
        self.assertEqual('1 is not 2', self.resolve(snippet, tmpl))
        snippet = {'Fn::Replace': [{'$var1': 1.3, '%var2%': 2.5}, '$var1 is not %var2%']}
        self.assertEqual('1.3 is not 2.5', self.resolve(snippet, tmpl))

    def test_replace_none_values(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::Replace': [{'$var1': None, '${var2}': None}, '"$var1" is "${var2}"']}
        self.assertEqual('"" is ""', self.resolve(snippet, tmpl))

    def test_replace_missing_key(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::Replace': [{'$var1': 'foo', 'var2': 'bar'}, '"$var1" is "${var3}"']}
        self.assertEqual('"foo" is "${var3}"', self.resolve(snippet, tmpl))

    def test_replace_param_values(self):
        env = environment.Environment({'foo': 'wibble'})
        tmpl = template.Template(parameter_template, env=env)
        stk = stack.Stack(self.ctx, 'test_stack', tmpl)
        snippet = {'Fn::Replace': [{'$var1': {'Ref': 'foo'}, '%var2%': {'Ref': 'blarg'}}, '$var1 is %var2%']}
        self.assertEqual('wibble is quux', self.resolve(snippet, tmpl, stk))

    def test_member_list2map_good(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::MemberListToMap': ['Name', 'Value', ['.member.0.Name=metric', '.member.0.Value=cpu', '.member.1.Name=size', '.member.1.Value=56']]}
        self.assertEqual({'metric': 'cpu', 'size': '56'}, self.resolve(snippet, tmpl))

    def test_member_list2map_good2(self):
        tmpl = template.Template(empty_template)
        snippet = {'Fn::MemberListToMap': ['Key', 'Value', ['.member.2.Key=metric', '.member.2.Value=cpu', '.member.5.Key=size', '.member.5.Value=56']]}
        self.assertEqual({'metric': 'cpu', 'size': '56'}, self.resolve(snippet, tmpl))

    def test_resource_facade(self):
        metadata_snippet = {'Fn::ResourceFacade': 'Metadata'}
        deletion_policy_snippet = {'Fn::ResourceFacade': 'DeletionPolicy'}
        update_policy_snippet = {'Fn::ResourceFacade': 'UpdatePolicy'}
        parent_resource = DummyClass()
        parent_resource.metadata_set({'foo': 'bar'})
        parent_resource.t = rsrc_defn.ResourceDefinition('parent', 'SomeType', deletion_policy=rsrc_defn.ResourceDefinition.RETAIN, update_policy={'blarg': 'wibble'})
        tmpl = copy.deepcopy(empty_template)
        tmpl['Resources'] = {'parent': {'Type': 'SomeType', 'DeletionPolicy': 'Retain', 'UpdatePolicy': {'blarg': 'wibble'}}}
        parent_resource.stack = stack.Stack(self.ctx, 'toplevel_stack', template.Template(tmpl))
        parent_resource.stack._resources = {'parent': parent_resource}
        stk = stack.Stack(self.ctx, 'test_stack', template.Template(empty_template), parent_resource='parent', owner_id=45)
        stk.set_parent_stack(parent_resource.stack)
        self.assertEqual({'foo': 'bar'}, self.resolve(metadata_snippet, stk.t, stk))
        self.assertEqual('Retain', self.resolve(deletion_policy_snippet, stk.t, stk))
        self.assertEqual({'blarg': 'wibble'}, self.resolve(update_policy_snippet, stk.t, stk))

    def test_resource_facade_function(self):
        deletion_policy_snippet = {'Fn::ResourceFacade': 'DeletionPolicy'}
        parent_resource = DummyClass()
        parent_resource.metadata_set({'foo': 'bar'})
        del_policy = cfn_funcs.Join(None, 'Fn::Join', ['eta', ['R', 'in']])
        parent_resource.t = rsrc_defn.ResourceDefinition('parent', 'SomeType', deletion_policy=del_policy)
        tmpl = copy.deepcopy(empty_template)
        tmpl['Resources'] = {'parent': {'Type': 'SomeType', 'DeletionPolicy': del_policy}}
        parent_resource.stack = stack.Stack(self.ctx, 'toplevel_stack', template.Template(tmpl))
        parent_resource.stack._resources = {'parent': parent_resource}
        stk = stack.Stack(self.ctx, 'test_stack', template.Template(empty_template), parent_resource='parent')
        stk.set_parent_stack(parent_resource.stack)
        self.assertEqual('Retain', self.resolve(deletion_policy_snippet, stk.t, stk))

    def test_resource_facade_invalid_arg(self):
        snippet = {'Fn::ResourceFacade': 'wibble'}
        stk = stack.Stack(self.ctx, 'test_stack', template.Template(empty_template))
        error = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, stk.t, stk)
        self.assertIn(next(iter(snippet)), str(error))

    def test_resource_facade_missing_deletion_policy(self):
        snippet = {'Fn::ResourceFacade': 'DeletionPolicy'}
        parent_resource = DummyClass()
        parent_resource.metadata_set({'foo': 'bar'})
        parent_resource.t = rsrc_defn.ResourceDefinition('parent', 'SomeType')
        tmpl = copy.deepcopy(empty_template)
        tmpl['Resources'] = {'parent': {'Type': 'SomeType'}}
        parent_resource.stack = stack.Stack(self.ctx, 'toplevel_stack', template.Template(tmpl))
        parent_resource.stack._resources = {'parent': parent_resource}
        stk = stack.Stack(self.ctx, 'test_stack', template.Template(empty_template), parent_resource='parent', owner_id=78)
        stk.set_parent_stack(parent_resource.stack)
        self.assertEqual('Delete', self.resolve(snippet, stk.t, stk))

    def test_prevent_parameters_access(self):
        expected_description = 'This can be accessed'
        tmpl = template.Template({'AWSTemplateFormatVersion': '2010-09-09', 'Description': expected_description, 'Parameters': {'foo': {'Type': 'String', 'Required': True}}})
        self.assertEqual(expected_description, tmpl['Description'])
        keyError = self.assertRaises(KeyError, tmpl.__getitem__, 'Parameters')
        self.assertIn('can not be accessed directly', str(keyError))

    def test_parameters_section_not_iterable(self):
        expected_description = 'This can be accessed'
        tmpl = template.Template({'AWSTemplateFormatVersion': '2010-09-09', 'Description': expected_description, 'Parameters': {'foo': {'Type': 'String', 'Required': True}}})
        self.assertEqual(expected_description, tmpl['Description'])
        self.assertNotIn('Parameters', tmpl.keys())

    def test_add_resource(self):
        cfn_tpl = template_format.parse('\n        AWSTemplateFormatVersion: 2010-09-09\n        Resources:\n          resource1:\n            Type: AWS::EC2::Instance\n            Properties:\n              property1: value1\n            Metadata:\n              foo: bar\n            DependsOn: dummy\n            DeletionPolicy: Retain\n            UpdatePolicy:\n              foo: bar\n          resource2:\n            Type: AWS::EC2::Instance\n          resource3:\n            Type: AWS::EC2::Instance\n            DependsOn:\n              - resource1\n              - dummy\n              - resource2\n        ')
        source = template.Template(cfn_tpl)
        empty = template.Template(copy.deepcopy(empty_template))
        stk = stack.Stack(self.ctx, 'test_stack', source)
        for rname, defn in sorted(source.resource_definitions(stk).items()):
            empty.add_resource(defn)
        expected = copy.deepcopy(cfn_tpl['Resources'])
        del expected['resource1']['DependsOn']
        expected['resource3']['DependsOn'] = ['resource1', 'resource2']
        self.assertEqual(expected, empty.t['Resources'])

    def test_add_output(self):
        cfn_tpl = template_format.parse('\n        AWSTemplateFormatVersion: 2010-09-09\n        Outputs:\n          output1:\n            Description: An output\n            Value: foo\n        ')
        source = template.Template(cfn_tpl)
        empty = template.Template(copy.deepcopy(empty_template))
        stk = stack.Stack(self.ctx, 'test_stack', source)
        for defn in source.outputs(stk).values():
            empty.add_output(defn)
        self.assertEqual(cfn_tpl['Outputs'], empty.t['Outputs'])

    def test_create_empty_template_default_version(self):
        empty_template = template.Template.create_empty_template()
        self.assertEqual(hot_t.HOTemplate20150430, empty_template.__class__)
        self.assertEqual({}, empty_template['parameter_groups'])
        self.assertEqual({}, empty_template['resources'])
        self.assertEqual({}, empty_template['outputs'])

    def test_create_empty_template_returns_correct_version(self):
        t = template_format.parse('\n            AWSTemplateFormatVersion: 2010-09-09\n            Parameters:\n            Resources:\n            Outputs:\n            ')
        aws_tmpl = template.Template(t)
        empty_template = template.Template.create_empty_template(version=aws_tmpl.version)
        self.assertEqual(aws_tmpl.__class__, empty_template.__class__)
        self.assertEqual({}, empty_template['Mappings'])
        self.assertEqual({}, empty_template['Resources'])
        self.assertEqual({}, empty_template['Outputs'])
        t = template_format.parse('\n            HeatTemplateFormatVersion: 2012-12-12\n            Parameters:\n            Resources:\n            Outputs:\n            ')
        heat_tmpl = template.Template(t)
        empty_template = template.Template.create_empty_template(version=heat_tmpl.version)
        self.assertEqual(heat_tmpl.__class__, empty_template.__class__)
        self.assertEqual({}, empty_template['Mappings'])
        self.assertEqual({}, empty_template['Resources'])
        self.assertEqual({}, empty_template['Outputs'])
        t = template_format.parse('\n            heat_template_version: 2015-04-30\n            parameter_groups:\n            resources:\n            outputs:\n            ')
        hot_tmpl = template.Template(t)
        empty_template = template.Template.create_empty_template(version=hot_tmpl.version)
        self.assertEqual(hot_tmpl.__class__, empty_template.__class__)
        self.assertEqual({}, empty_template['parameter_groups'])
        self.assertEqual({}, empty_template['resources'])
        self.assertEqual({}, empty_template['outputs'])

    def test_create_empty_template_from_another_template(self):
        res_param_template = template_format.parse('{\n          "HeatTemplateFormatVersion" : "2012-12-12",\n          "Parameters" : {\n            "foo" : { "Type" : "String" },\n            "blarg" : { "Type" : "String", "Default": "quux" }\n          },\n          "Resources" : {\n            "foo" : { "Type" : "GenericResourceType" },\n            "blarg" : { "Type" : "GenericResourceType" }\n          }\n        }')
        env = environment.Environment({'foo': 'bar'})
        hot_tmpl = template.Template(res_param_template, env)
        empty_template = template.Template.create_empty_template(from_template=hot_tmpl)
        self.assertEqual({}, empty_template['Resources'])
        self.assertEqual(hot_tmpl.env, empty_template.env)