import collections
import json
import os
from unittest import mock
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources import template_resource
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class ProviderTemplateTest(common.HeatTestCase):

    def setUp(self):
        super(ProviderTemplateTest, self).setUp()
        resource._register_class('myCloud::ResourceType', MyCloudResource)

    def test_get_os_empty_registry(self):
        env_str = {'resource_registry': {}}
        env = environment.Environment(env_str)
        cls = env.get_class('GenericResourceType', 'fred')
        self.assertEqual(generic_rsrc.GenericResource, cls)

    def test_get_mine_global_map(self):
        env_str = {'resource_registry': {'OS::*': 'myCloud::*'}}
        env = environment.Environment(env_str)
        cls = env.get_class('OS::ResourceType', 'fred')
        self.assertEqual(MyCloudResource, cls)

    def test_get_mine_type_map(self):
        env_str = {'resource_registry': {'OS::ResourceType': 'myCloud::ResourceType'}}
        env = environment.Environment(env_str)
        cls = env.get_class('OS::ResourceType', 'fred')
        self.assertEqual(MyCloudResource, cls)

    def test_get_mine_resource_map(self):
        env_str = {'resource_registry': {'resources': {'fred': {'OS::ResourceType': 'myCloud::ResourceType'}}}}
        env = environment.Environment(env_str)
        cls = env.get_class('OS::ResourceType', 'fred')
        self.assertEqual(MyCloudResource, cls)

    def test_get_os_no_match(self):
        env_str = {'resource_registry': {'resources': {'jerry': {'OS::ResourceType': 'myCloud::ResourceType'}}}}
        env = environment.Environment(env_str)
        cls = env.get_class('GenericResourceType', 'fred')
        self.assertEqual(generic_rsrc.GenericResource, cls)

    def test_to_parameters(self):
        """Tests property conversion to parameter values."""
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'String'}, 'AList': {'Type': 'CommaDelimitedList'}, 'MemList': {'Type': 'CommaDelimitedList'}, 'ListEmpty': {'Type': 'CommaDelimitedList'}, 'ANum': {'Type': 'Number'}, 'AMap': {'Type': 'Json'}}, 'Outputs': {'Foo': {'Value': 'bar'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            attributes_schema = {'Foo': attributes.Schema('A test attribute')}
            properties_schema = {'Foo': {'Type': 'String'}, 'AList': {'Type': 'List'}, 'MemList': {'Type': 'List'}, 'ListEmpty': {'Type': 'List'}, 'ANum': {'Type': 'Number'}, 'AMap': {'Type': 'Map'}}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        map_prop_val = {'key1': 'val1', 'key2': ['lval1', 'lval2', 'lval3'], 'key3': {'key4': 4, 'key5': False}}
        prop_vals = {'Foo': 'Bar', 'AList': ['one', 'two', 'three'], 'MemList': [collections.OrderedDict([('key', 'name'), ('value', 'three')]), collections.OrderedDict([('key', 'name'), ('value', 'four')])], 'ListEmpty': [], 'ANum': 5, 'AMap': map_prop_val}
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', prop_vals)
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        temp_res.validate()
        converted_params = temp_res.child_params()
        self.assertTrue(converted_params)
        for key in DummyResource.properties_schema:
            self.assertIn(key, converted_params)
        self.assertEqual('Bar', converted_params.get('Foo'))
        self.assertEqual('one,two,three', converted_params.get('AList'))
        mem_exp = '.member.0.key=name,.member.0.value=three,.member.1.key=name,.member.1.value=four'
        self.assertEqual(sorted(mem_exp.split(',')), sorted(converted_params.get('MemList').split(',')))
        self.assertEqual(5, converted_params.get('ANum'))
        self.assertEqual(map_prop_val, converted_params.get('AMap'))
        with mock.patch.object(properties.Properties, 'get_user_value') as m_get:
            m_get.side_effect = ValueError('boom')
            temp_res.action = temp_res.INIT
            converted_params = temp_res.child_params()
            for key in DummyResource.properties_schema:
                self.assertIn(key, converted_params)
            self.assertEqual({}, converted_params['AMap'])
            self.assertEqual(0, converted_params['ANum'])
            temp_res.action = temp_res.CREATE
            self.assertRaises(ValueError, temp_res.child_params)

    def test_attributes_extra(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Outputs': {'Foo': {'Value': 'bar'}, 'Blarg': {'Value': 'wibble'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            attributes_schema = {'Foo': attributes.Schema('A test attribute')}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertIsNone(temp_res.validate())

    def test_attributes_missing_based_on_class(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Outputs': {'Blarg': {'Value': 'wibble'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            attributes_schema = {'Foo': attributes.Schema('A test attribute')}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertRaises(exception.StackValidationFailed, temp_res.validate)

    def test_attributes_missing_no_class(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Outputs': {'Blarg': {'Value': 'wibble'}}}
        files = {'test_resource.template': json.dumps(provider)}
        env = environment.Environment()
        env.load({'resource_registry': {'DummyResource2': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource2')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        temp_res.resource_id = 'dummy_id'
        temp_res.nested_identifier = mock.Mock()
        temp_res.nested_identifier.return_value = {'foo': 'bar'}
        temp_res._rpc_client = mock.MagicMock()
        output = {'outputs': [{'output_key': 'Blarg', 'output_value': 'fluffy'}]}
        temp_res._rpc_client.show_stack.return_value = [output]
        self.assertRaises(exception.InvalidTemplateAttribute, temp_res.FnGetAtt, 'Foo')

    def test_attributes_not_parsable(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Outputs': {'Foo': {'Value': 'bar'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            support_status = support.SupportStatus()
            properties_schema = {}
            attributes_schema = {'Foo': attributes.Schema('A test attribute')}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        temp_res.resource_id = 'dummy_id'
        temp_res.nested_identifier = mock.Mock()
        temp_res.nested_identifier.return_value = {'foo': 'bar'}
        temp_res._rpc_client = mock.MagicMock()
        output = {'outputs': [{'output_key': 'Foo', 'output_value': None, 'output_error': 'it is all bad'}]}
        temp_res._rpc_client.show_stack.return_value = [output]
        temp_res._rpc_client.list_stack_resources.return_value = []
        self.assertIsNone(temp_res.validate())
        self.assertRaises(exception.TemplateOutputError, temp_res.FnGetAtt, 'Foo')

    def test_properties_normal(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'String'}, 'Blarg': {'Type': 'String', 'Default': 'wibble'}}}
        files = {'test_resource.template': json.dumps(provider)}
        env = environment.Environment()
        env.load({'resource_registry': {'ResourceWithRequiredPropsAndEmptyAttrs': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'ResourceWithRequiredPropsAndEmptyAttrs', {'Foo': 'bar'})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertIsNone(temp_res.validate())

    def test_properties_missing(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Blarg': {'Type': 'String', 'Default': 'wibble'}}}
        files = {'test_resource.template': json.dumps(provider)}
        env = environment.Environment()
        env.load({'resource_registry': {'ResourceWithRequiredPropsAndEmptyAttrs': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'ResourceWithRequiredPropsAndEmptyAttrs')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertRaises(exception.StackValidationFailed, temp_res.validate)

    def test_properties_extra_required(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Blarg': {'Type': 'String'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(object):
            support_status = support.SupportStatus()
            properties_schema = {}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Blarg': 'wibble'})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertRaises(exception.StackValidationFailed, temp_res.validate)

    def test_properties_type_mismatch(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'String'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            support_status = support.SupportStatus()
            properties_schema = {'Foo': properties.Schema(properties.Schema.MAP)}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12'}, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Foo': 'bar'})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        ex = self.assertRaises(exception.StackValidationFailed, temp_res.validate)
        self.assertEqual('Property Foo type mismatch between facade DummyResource (Map) and provider (String)', str(ex))

    def test_properties_list_with_none(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'CommaDelimitedList'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            support_status = support.SupportStatus()
            properties_schema = {'Foo': properties.Schema(properties.Schema.LIST)}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12'}, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Foo': [None, 'test', None]})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertIsNone(temp_res.validate())
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Foo': [None, None, None]})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertIsNone(temp_res.validate())

    def test_properties_type_match(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Length': {'Type': 'Number'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            support_status = support.SupportStatus()
            properties_schema = {'Length': properties.Schema(properties.Schema.INTEGER)}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12'}, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Length': 10})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertIsNone(temp_res.validate())

    def test_boolean_type_provider(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'Boolean'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            support_status = support.SupportStatus()
            properties_schema = {'Foo': properties.Schema(properties.Schema.BOOLEAN)}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12'}, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Foo': 'False'})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertIsNone(temp_res.validate())

    def test_resource_info_general(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'Boolean'}}}
        files = {'test_resource.template': json.dumps(provider), 'foo.template': json.dumps(provider)}

        class DummyResource(generic_rsrc.GenericResource):
            properties_schema = {'Foo': properties.Schema(properties.Schema.BOOLEAN)}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template', 'resources': {'foo': 'foo.template'}}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12'}, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Foo': 'False'})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertEqual('test_resource.template', temp_res.template_url)

    def test_resource_info_special(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'Boolean'}}}
        files = {'test_resource.template': json.dumps(provider), 'foo.template': json.dumps(provider)}

        class DummyResource(object):
            support_status = support.SupportStatus()
            properties_schema = {'Foo': properties.Schema(properties.Schema.BOOLEAN)}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template', 'resources': {'foo': {'DummyResource': 'foo.template'}}}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12'}, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('foo', 'DummyResource', {'Foo': 'False'})
        temp_res = template_resource.TemplateResource('foo', definition, stack)
        self.assertEqual('foo.template', temp_res.template_url)

    def test_get_error_for_invalid_template_name(self):
        env_str = {'resource_registry': {'resources': {'fred': {'OS::ResourceType': 'some_magic.yaml'}}}}
        env = environment.Environment(env_str)
        ex = self.assertRaises(exception.NotFound, env.get_class, 'OS::ResourceType', 'fred')
        self.assertIn('Could not fetch remote template "some_magic.yaml"', str(ex))

    def test_metadata_update_called(self):
        provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'Boolean'}}}
        files = {'test_resource.template': json.dumps(provider)}

        class DummyResource(object):
            support_status = support.SupportStatus()
            properties_schema = {'Foo': properties.Schema(properties.Schema.BOOLEAN)}
            attributes_schema = {}
        env = environment.Environment()
        resource._register_class('DummyResource', DummyResource)
        env.load({'resource_registry': {'DummyResource': 'test_resource.template'}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12'}, files=files, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'DummyResource', {'Foo': 'False'})
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        temp_res.metadata_set = mock.Mock()
        temp_res.metadata_update()
        temp_res.metadata_set.assert_called_once_with({})

    def test_get_template_resource_class(self):
        test_templ_name = 'file:///etc/heatr/frodo.yaml'
        minimal_temp = json.dumps({'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {}, 'Resources': {}})
        mock_get = self.patchobject(urlfetch, 'get', return_value=minimal_temp)
        env_str = {'resource_registry': {'resources': {'fred': {'OS::ResourceType': test_templ_name}}}}
        global_env = environment.Environment({}, user_env=False)
        global_env.load(env_str)
        with mock.patch('heat.engine.resources._environment', global_env):
            env = environment.Environment({})
        cls = env.get_class('OS::ResourceType', 'fred')
        self.assertNotEqual(template_resource.TemplateResource, cls)
        self.assertTrue(issubclass(cls, template_resource.TemplateResource))
        self.assertTrue(hasattr(cls, 'properties_schema'))
        self.assertTrue(hasattr(cls, 'attributes_schema'))
        mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('file',))

    def test_template_as_resource(self):
        """Test that resulting resource has the right prop and attrib schema.

        Note that this test requires the Wordpress_Single_Instance.yaml
        template in the templates directory since we want to test using a
        non-trivial template.
        """
        test_templ_name = 'WordPress_Single_Instance.yaml'
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates', test_templ_name)
        self.assertIn(test_templ_name, os.listdir(os.path.dirname(path)))
        with open(path) as test_templ_file:
            test_templ = test_templ_file.read()
        self.assertTrue(test_templ, 'Empty test template')
        mock_get = self.patchobject(urlfetch, 'get', return_value=test_templ)
        parsed_test_templ = template_format.parse(test_templ)
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template), stack_id=str(uuid.uuid4()))
        properties = {'KeyName': 'mykeyname', 'DBName': 'wordpress1', 'DBUsername': 'wpdbuser', 'DBPassword': 'wpdbpass', 'DBRootPassword': 'wpdbrootpass', 'LinuxDistribution': 'U10'}
        definition = rsrc_defn.ResourceDefinition('test_templ_resource', test_templ_name, properties)
        templ_resource = resource.Resource('test_templ_resource', definition, stack)
        self.assertIsInstance(templ_resource, template_resource.TemplateResource)
        for prop in parsed_test_templ.get('Parameters', {}):
            self.assertIn(prop, templ_resource.properties)
        for attrib in parsed_test_templ.get('Outputs', {}):
            self.assertIn(attrib, templ_resource.attributes)
        for k, v in properties.items():
            self.assertEqual(v, templ_resource.properties[k])
        self.assertEqual({'WordPress_Single_Instance.yaml': 'WordPress_Single_Instance.yaml', 'resources': {}}, stack.env.user_env_as_dict()['resource_registry'])
        self.assertNotIn('WordPress_Single_Instance.yaml', resources.global_env().registry._registry)
        mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('http', 'https'))

    def test_persisted_unregistered_provider_templates(self):
        """Test that templates are registered correctly.

        Test that templates persisted in the database prior to
        https://review.opendev.org/#/c/79953/1 are registered correctly.
        """
        env = {'resource_registry': {'http://example.com/test.template': None, 'resources': {}}}
        environment.Environment(env=env)

    def test_system_template_retrieve_by_file(self):
        g_env = resources.global_env()
        test_templ_name = 'file:///etc/heatr/frodo.yaml'
        g_env.load({'resource_registry': {'Test::Frodo': test_templ_name}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template), stack_id=str(uuid.uuid4()))
        minimal_temp = json.dumps({'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {}, 'Resources': {}})
        mock_get = self.patchobject(urlfetch, 'get', return_value=minimal_temp)
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Frodo')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertIsNone(temp_res.validate())
        mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('http', 'https', 'file'))

    def test_user_template_not_retrieved_by_file(self):
        env = environment.Environment()
        test_templ_name = 'file:///etc/heatr/flippy.yaml'
        env.load({'resource_registry': {'Test::Flippy': test_templ_name}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Flippy')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertRaises(exception.StackValidationFailed, temp_res.validate)

    def test_system_template_retrieve_fail(self):
        g_env = resources.global_env()
        test_templ_name = 'file:///etc/heatr/frodo.yaml'
        g_env.load({'resource_registry': {'Test::Frodo': test_templ_name}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template), stack_id=str(uuid.uuid4()))
        mock_get = self.patchobject(urlfetch, 'get', side_effect=urlfetch.URLFetchError(_('Failed to retrieve template')))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Frodo')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertRaises(exception.StackValidationFailed, temp_res.validate)
        mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('http', 'https', 'file'))

    def test_user_template_retrieve_fail(self):
        env = environment.Environment()
        test_templ_name = 'http://heatr/noexist.yaml'
        env.load({'resource_registry': {'Test::Flippy': test_templ_name}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=env), stack_id=str(uuid.uuid4()))
        mock_get = self.patchobject(urlfetch, 'get', side_effect=urlfetch.URLFetchError(_('Failed to retrieve template')))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Flippy')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertRaises(exception.StackValidationFailed, temp_res.validate)
        mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('http', 'https'))

    def test_user_template_retrieve_fail_ext(self):
        env = environment.Environment()
        test_templ_name = 'http://heatr/letter_to_granny.docx'
        env.load({'resource_registry': {'Test::Flippy': test_templ_name}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=env), stack_id=str(uuid.uuid4()))
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Flippy')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        self.assertRaises(exception.StackValidationFailed, temp_res.validate)

    def test_incorrect_template_provided_with_url(self):
        wrong_template = '\n         <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb#\n        '
        env = environment.Environment()
        test_templ_name = 'http://heatr/bad_tmpl.yaml'
        env.load({'resource_registry': {'Test::Tmpl': test_templ_name}})
        stack = parser.Stack(utils.dummy_context(), 'test_stack', template.Template(empty_template, env=env), stack_id=str(uuid.uuid4()))
        mock_get = self.patchobject(urlfetch, 'get', return_value=wrong_template)
        definition = rsrc_defn.ResourceDefinition('test_t_res', 'Test::Tmpl')
        temp_res = template_resource.TemplateResource('test_t_res', definition, stack)
        err = self.assertRaises(exception.StackValidationFailed, temp_res.validate)
        self.assertIn('Error parsing template http://heatr/bad_tmpl.yaml', str(err))
        mock_get.assert_called_once_with(test_templ_name, allowed_schemes=('http', 'https'))