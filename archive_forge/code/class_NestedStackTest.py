from unittest import mock
from oslo_config import cfg
from requests import exceptions
import yaml
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import api
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.cfn import stack as stack_res
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class NestedStackTest(common.HeatTestCase):
    test_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nResources:\n  the_nested:\n    Type: AWS::CloudFormation::Stack\n    Properties:\n      TemplateURL: https://server.test/the.template\n      Parameters:\n        KeyName: foo\n"
    nested_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  KeyName:\n    Type: String\nOutputs:\n  Foo:\n    Value: bar\n"
    update_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  KeyName:\n    Type: String\nOutputs:\n  Bar:\n    Value: foo\n"

    def setUp(self):
        super(NestedStackTest, self).setUp()
        self.patchobject(urlfetch, 'get')

    def validate_stack(self, template):
        t = template_format.parse(template)
        stack = self.parse_stack(t)
        res = stack.validate()
        self.assertIsNone(res)
        return stack

    def parse_stack(self, t, data=None):
        ctx = utils.dummy_context('test_username', 'aaaa', 'password')
        stack_name = 'test_stack'
        tmpl = template.Template(t)
        stack = parser.Stack(ctx, stack_name, tmpl, adopt_stack_data=data)
        stack.store()
        return stack

    @mock.patch.object(parser.Stack, 'total_resources')
    def test_nested_stack_three_deep(self, tr):
        root_template = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth1.template'\n"
        depth1_template = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth2.template'\n"
        depth2_template = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth3.template'\n            Parameters:\n                KeyName: foo\n"
        urlfetch.get.side_effect = [depth1_template, depth2_template, self.nested_template]
        tr.return_value = 2
        self.validate_stack(root_template)
        calls = [mock.call('https://server.test/depth1.template'), mock.call('https://server.test/depth2.template'), mock.call('https://server.test/depth3.template')]
        urlfetch.get.assert_has_calls(calls)

    @mock.patch.object(parser.Stack, 'total_resources')
    def test_nested_stack_six_deep(self, tr):
        tmpl = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth%i.template'\n"
        root_template = tmpl % 1
        depth1_template = tmpl % 2
        depth2_template = tmpl % 3
        depth3_template = tmpl % 4
        depth4_template = tmpl % 5
        depth5_template = tmpl % 6
        depth5_template += '\n            Parameters:\n                KeyName: foo\n'
        urlfetch.get.side_effect = [depth1_template, depth2_template, depth3_template, depth4_template, depth5_template, self.nested_template]
        tr.return_value = 5
        t = template_format.parse(root_template)
        stack = self.parse_stack(t)
        stack['Nested'].root_stack_id = '1234'
        res = self.assertRaises(exception.StackValidationFailed, stack.validate)
        self.assertIn('Recursion depth exceeds', str(res))
        calls = [mock.call('https://server.test/depth1.template'), mock.call('https://server.test/depth2.template'), mock.call('https://server.test/depth3.template'), mock.call('https://server.test/depth4.template'), mock.call('https://server.test/depth5.template'), mock.call('https://server.test/depth6.template')]
        urlfetch.get.assert_has_calls(calls)

    def test_nested_stack_four_wide(self):
        root_template = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth1.template'\n            Parameters:\n                KeyName: foo\n    Nested2:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth2.template'\n            Parameters:\n                KeyName: foo\n    Nested3:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth3.template'\n            Parameters:\n                KeyName: foo\n    Nested4:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/depth4.template'\n            Parameters:\n                KeyName: foo\n"
        urlfetch.get.return_value = self.nested_template
        self.validate_stack(root_template)
        calls = [mock.call('https://server.test/depth1.template'), mock.call('https://server.test/depth2.template'), mock.call('https://server.test/depth3.template'), mock.call('https://server.test/depth4.template')]
        urlfetch.get.assert_has_calls(calls, any_order=True)

    @mock.patch.object(parser.Stack, 'total_resources')
    def test_nested_stack_infinite_recursion(self, tr):
        tmpl = "\nHeatTemplateFormatVersion: 2012-12-12\nResources:\n    Nested:\n        Type: AWS::CloudFormation::Stack\n        Properties:\n            TemplateURL: 'https://server.test/the.template'\n"
        urlfetch.get.return_value = tmpl
        t = template_format.parse(tmpl)
        stack = self.parse_stack(t)
        stack['Nested'].root_stack_id = '1234'
        tr.return_value = 2
        res = self.assertRaises(exception.StackValidationFailed, stack.validate)
        self.assertIn('Recursion depth exceeds', str(res))
        expected_count = cfg.CONF.get('max_nested_stack_depth') + 1
        self.assertEqual(expected_count, urlfetch.get.call_count)

    def test_child_params(self):
        t = template_format.parse(self.test_template)
        stack = self.parse_stack(t)
        nested_stack = stack['the_nested']
        nested_stack.properties.data[nested_stack.PARAMETERS] = {'foo': 'bar'}
        self.assertEqual({'foo': 'bar'}, nested_stack.child_params())

    def test_child_template_when_file_is_fetched(self):
        urlfetch.get.return_value = 'template_file'
        t = template_format.parse(self.test_template)
        stack = self.parse_stack(t)
        nested_stack = stack['the_nested']
        with mock.patch('heat.common.template_format.parse') as mock_parse:
            mock_parse.return_value = 'child_template'
            self.assertEqual('child_template', nested_stack.child_template())
            mock_parse.assert_called_once_with('template_file', 'https://server.test/the.template')

    def test_child_template_when_fetching_file_fails(self):
        urlfetch.get.side_effect = exceptions.RequestException()
        t = template_format.parse(self.test_template)
        stack = self.parse_stack(t)
        nested_stack = stack['the_nested']
        self.assertRaises(ValueError, nested_stack.child_template)

    def test_child_template_when_io_error(self):
        msg = 'Failed to retrieve template'
        urlfetch.get.side_effect = urlfetch.URLFetchError(msg)
        t = template_format.parse(self.test_template)
        stack = self.parse_stack(t)
        nested_stack = stack['the_nested']
        self.assertRaises(ValueError, nested_stack.child_template)

    def test_refid(self):
        t = template_format.parse(self.test_template)
        stack = self.parse_stack(t)
        nested_stack = stack['the_nested']
        self.assertEqual('the_nested', nested_stack.FnGetRefId())

    def test_refid_convergence_cache_data(self):
        t = template_format.parse(self.test_template)
        tmpl = template.Template(t)
        ctx = utils.dummy_context()
        cache_data = {'the_nested': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'the_nested_convg_mock'})}
        stack = parser.Stack(ctx, 'test_stack', tmpl, cache_data=cache_data)
        nested_stack = stack.defn['the_nested']
        self.assertEqual('the_nested_convg_mock', nested_stack.FnGetRefId())

    def test_get_attribute(self):
        tmpl = template_format.parse(self.test_template)
        ctx = utils.dummy_context('test_username', 'aaaa', 'password')
        stack = parser.Stack(ctx, 'test', template.Template(tmpl))
        stack.store()
        stack_res = stack['the_nested']
        stack_res.store()
        nested_t = template_format.parse(self.nested_template)
        nested_t['Parameters']['KeyName']['Default'] = 'Key'
        nested_stack = parser.Stack(ctx, 'test_nested', template.Template(nested_t))
        nested_stack.store()
        stack_res._rpc_client = mock.MagicMock()
        stack_res._rpc_client.show_stack.return_value = [api.format_stack(nested_stack)]
        stack_res.nested_identifier = mock.Mock()
        stack_res.nested_identifier.return_value = {'foo': 'bar'}
        self.assertEqual('bar', stack_res.FnGetAtt('Outputs.Foo'))