from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
class ParametersTest(ParametersBase):

    def test_pseudo_params(self):
        stack_name = 'test_stack'
        params = self.new_parameters(stack_name, {'Parameters': {}})
        self.assertEqual('test_stack', params['AWS::StackName'])
        self.assertEqual('arn:openstack:heat:::stacks/{0}/{1}'.format(stack_name, 'None'), params['AWS::StackId'])
        self.assertIn('AWS::Region', params)

    def test_pseudo_param_stackid(self):
        stack_name = 'test_stack'
        params = self.new_parameters(stack_name, {'Parameters': {}}, stack_id='abc123')
        self.assertEqual('arn:openstack:heat:::stacks/{0}/{1}'.format(stack_name, 'abc123'), params['AWS::StackId'])
        stack_identifier = identifier.HeatIdentifier('', '', 'def456')
        params.set_stack_id(stack_identifier)
        self.assertEqual(stack_identifier.arn(), params['AWS::StackId'])

    def test_schema_invariance(self):
        params1 = self.new_parameters('test', params_schema, {'User': 'foo', 'Defaulted': 'wibble'})
        self.assertEqual('wibble', params1['Defaulted'])
        params2 = self.new_parameters('test', params_schema, {'User': 'foo'})
        self.assertEqual('foobar', params2['Defaulted'])

    def test_to_dict(self):
        template = {'Parameters': {'Foo': {'Type': 'String'}, 'Bar': {'Type': 'Number', 'Default': '42'}}}
        params = self.new_parameters('test_params', template, {'Foo': 'foo'})
        as_dict = dict(params)
        self.assertEqual('foo', as_dict['Foo'])
        self.assertEqual(42, as_dict['Bar'])
        self.assertEqual('test_params', as_dict['AWS::StackName'])
        self.assertIn('AWS::Region', as_dict)

    def test_map(self):
        template = {'Parameters': {'Foo': {'Type': 'String'}, 'Bar': {'Type': 'Number', 'Default': '42'}}}
        params = self.new_parameters('test_params', template, {'Foo': 'foo'})
        expected = {'Foo': False, 'Bar': True, 'AWS::Region': True, 'AWS::StackId': True, 'AWS::StackName': True}
        self.assertEqual(expected, params.map(lambda p: p.has_default()))

    def test_map_str(self):
        template = {'Parameters': {'Foo': {'Type': 'String'}, 'Bar': {'Type': 'Number'}, 'Uni': {'Type': 'String'}}}
        stack_name = 'test_params'
        params = self.new_parameters(stack_name, template, {'Foo': 'foo', 'Bar': '42', 'Uni': u'test♥'})
        expected = {'Foo': 'foo', 'Bar': '42', 'Uni': b'test\xe2\x99\xa5', 'AWS::Region': 'ap-southeast-1', 'AWS::StackId': 'arn:openstack:heat:::stacks/{0}/{1}'.format(stack_name, 'None'), 'AWS::StackName': 'test_params'}
        mapped_params = params.map(str)
        mapped_params['Uni'] = mapped_params['Uni'].encode('utf-8')
        self.assertEqual(expected, mapped_params)

    def test_unknown_params(self):
        user_params = {'Foo': 'wibble'}
        self.assertRaises(exception.UnknownUserParameter, self.new_parameters, 'test', params_schema, user_params)

    def test_missing_params(self):
        user_params = {}
        self.assertRaises(exception.UserParameterMissing, self.new_parameters, 'test', params_schema, user_params)

    def test_missing_attribute_params(self):
        params = {'Parameters': {'Foo': {'Type': 'String'}, 'NoAttr': 'No attribute.', 'Bar': {'Type': 'Number', 'Default': '1'}}}
        self.assertRaises(exception.InvalidSchemaError, self.new_parameters, 'test', params)