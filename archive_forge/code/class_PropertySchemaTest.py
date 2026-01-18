from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
class PropertySchemaTest(common.HeatTestCase):

    def test_schema_all(self):
        d = {'type': 'string', 'description': 'A string', 'default': 'wibble', 'required': False, 'update_allowed': False, 'immutable': False, 'constraints': [{'length': {'min': 4, 'max': 8}}]}
        s = properties.Schema(properties.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Length(4, 8)])
        self.assertEqual(d, dict(s))

    def test_schema_list_schema(self):
        d = {'type': 'list', 'description': 'A list', 'schema': {'*': {'type': 'string', 'description': 'A string', 'default': 'wibble', 'required': False, 'update_allowed': False, 'immutable': False, 'constraints': [{'length': {'min': 4, 'max': 8}}]}}, 'required': False, 'update_allowed': False, 'immutable': False}
        s = properties.Schema(properties.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Length(4, 8)])
        ls = properties.Schema(properties.Schema.LIST, 'A list', schema=s)
        self.assertEqual(d, dict(ls))

    def test_schema_map_schema(self):
        d = {'type': 'map', 'description': 'A map', 'schema': {'Foo': {'type': 'string', 'description': 'A string', 'default': 'wibble', 'required': False, 'update_allowed': False, 'immutable': False, 'constraints': [{'length': {'min': 4, 'max': 8}}]}}, 'required': False, 'update_allowed': False, 'immutable': False}
        s = properties.Schema(properties.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Length(4, 8)])
        m = properties.Schema(properties.Schema.MAP, 'A map', schema={'Foo': s})
        self.assertEqual(d, dict(m))

    def test_schema_nested_schema(self):
        d = {'type': 'list', 'description': 'A list', 'schema': {'*': {'type': 'map', 'description': 'A map', 'schema': {'Foo': {'type': 'string', 'description': 'A string', 'default': 'wibble', 'required': False, 'update_allowed': False, 'immutable': False, 'constraints': [{'length': {'min': 4, 'max': 8}}]}}, 'required': False, 'update_allowed': False, 'immutable': False}}, 'required': False, 'update_allowed': False, 'immutable': False}
        s = properties.Schema(properties.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Length(4, 8)])
        m = properties.Schema(properties.Schema.MAP, 'A map', schema={'Foo': s})
        ls = properties.Schema(properties.Schema.LIST, 'A list', schema=m)
        self.assertEqual(d, dict(ls))

    def test_all_resource_schemata(self):
        for resource_type in resources.global_env().get_types():
            for schema in getattr(resource_type, 'properties_schema', {}).values():
                properties.Schema.from_legacy(schema)

    def test_from_legacy_idempotency(self):
        s = properties.Schema(properties.Schema.STRING)
        self.assertTrue(properties.Schema.from_legacy(s) is s)

    def test_from_legacy_minimal_string(self):
        s = properties.Schema.from_legacy({'Type': 'String'})
        self.assertEqual(properties.Schema.STRING, s.type)
        self.assertIsNone(s.description)
        self.assertIsNone(s.default)
        self.assertFalse(s.required)
        self.assertEqual(0, len(s.constraints))

    def test_from_legacy_string(self):
        s = properties.Schema.from_legacy({'Type': 'String', 'Description': 'a string', 'Default': 'wibble', 'Implemented': False, 'MinLength': 4, 'MaxLength': 8, 'AllowedValues': ['blarg', 'wibble'], 'AllowedPattern': '[a-z]*'})
        self.assertEqual(properties.Schema.STRING, s.type)
        self.assertEqual('a string', s.description)
        self.assertEqual('wibble', s.default)
        self.assertFalse(s.required)
        self.assertEqual(3, len(s.constraints))
        self.assertFalse(s.immutable)

    def test_from_legacy_min_length(self):
        s = properties.Schema.from_legacy({'Type': 'String', 'MinLength': 4})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Length)
        self.assertEqual(4, c.min)
        self.assertIsNone(c.max)

    def test_from_legacy_max_length(self):
        s = properties.Schema.from_legacy({'Type': 'String', 'MaxLength': 8})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Length)
        self.assertIsNone(c.min)
        self.assertEqual(8, c.max)

    def test_from_legacy_minmax_length(self):
        s = properties.Schema.from_legacy({'Type': 'String', 'MinLength': 4, 'MaxLength': 8})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Length)
        self.assertEqual(4, c.min)
        self.assertEqual(8, c.max)

    def test_from_legacy_minmax_string_length(self):
        s = properties.Schema.from_legacy({'Type': 'String', 'MinLength': '4', 'MaxLength': '8'})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Length)
        self.assertEqual(4, c.min)
        self.assertEqual(8, c.max)

    def test_from_legacy_min_value(self):
        s = properties.Schema.from_legacy({'Type': 'Integer', 'MinValue': 4})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Range)
        self.assertEqual(4, c.min)
        self.assertIsNone(c.max)

    def test_from_legacy_max_value(self):
        s = properties.Schema.from_legacy({'Type': 'Integer', 'MaxValue': 8})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Range)
        self.assertIsNone(c.min)
        self.assertEqual(8, c.max)

    def test_from_legacy_minmax_value(self):
        s = properties.Schema.from_legacy({'Type': 'Integer', 'MinValue': 4, 'MaxValue': 8})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Range)
        self.assertEqual(4, c.min)
        self.assertEqual(8, c.max)

    def test_from_legacy_minmax_string_value(self):
        s = properties.Schema.from_legacy({'Type': 'Integer', 'MinValue': '4', 'MaxValue': '8'})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.Range)
        self.assertEqual(4, c.min)
        self.assertEqual(8, c.max)

    def test_from_legacy_allowed_values(self):
        s = properties.Schema.from_legacy({'Type': 'String', 'AllowedValues': ['blarg', 'wibble']})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.AllowedValues)
        self.assertEqual(('blarg', 'wibble'), c.allowed)

    def test_from_legacy_allowed_pattern(self):
        s = properties.Schema.from_legacy({'Type': 'String', 'AllowedPattern': '[a-z]*'})
        self.assertEqual(1, len(s.constraints))
        c = s.constraints[0]
        self.assertIsInstance(c, constraints.AllowedPattern)
        self.assertEqual('[a-z]*', c.pattern)

    def test_from_legacy_list(self):
        ls = properties.Schema.from_legacy({'Type': 'List', 'Default': ['wibble'], 'Schema': {'Type': 'String', 'Default': 'wibble', 'MaxLength': 8}})
        self.assertEqual(properties.Schema.LIST, ls.type)
        self.assertEqual(['wibble'], ls.default)
        ss = ls.schema[0]
        self.assertEqual(properties.Schema.STRING, ss.type)
        self.assertEqual('wibble', ss.default)

    def test_from_legacy_map(self):
        ls = properties.Schema.from_legacy({'Type': 'Map', 'Schema': {'foo': {'Type': 'String', 'Default': 'wibble'}}})
        self.assertEqual(properties.Schema.MAP, ls.type)
        ss = ls.schema['foo']
        self.assertEqual(properties.Schema.STRING, ss.type)
        self.assertEqual('wibble', ss.default)

    def test_from_legacy_invalid_key(self):
        self.assertRaises(exception.InvalidSchemaError, properties.Schema.from_legacy, {'Type': 'String', 'Foo': 'Bar'})

    def test_from_string_param(self):
        description = 'WebServer EC2 instance type'
        allowed_values = ['t1.micro', 'm1.small', 'm1.large', 'm1.xlarge', 'm2.xlarge', 'm2.2xlarge', 'm2.4xlarge', 'c1.medium', 'c1.xlarge', 'cc1.4xlarge']
        constraint_desc = 'Must be a valid EC2 instance type.'
        param = parameters.Schema.from_dict('name', {'Type': 'String', 'Description': description, 'Default': 'm1.large', 'AllowedValues': allowed_values, 'ConstraintDescription': constraint_desc})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.STRING, schema.type)
        self.assertEqual(description, schema.description)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertEqual(1, len(schema.constraints))
        allowed_constraint = schema.constraints[0]
        self.assertEqual(tuple(allowed_values), allowed_constraint.allowed)
        self.assertEqual(constraint_desc, allowed_constraint.description)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_string_allowed_pattern(self):
        description = 'WebServer EC2 instance type'
        allowed_pattern = '[A-Za-z0-9.]*'
        constraint_desc = 'Must contain only alphanumeric characters.'
        param = parameters.Schema.from_dict('name', {'Type': 'String', 'Description': description, 'Default': 'm1.large', 'AllowedPattern': allowed_pattern, 'ConstraintDescription': constraint_desc})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.STRING, schema.type)
        self.assertEqual(description, schema.description)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertEqual(1, len(schema.constraints))
        allowed_constraint = schema.constraints[0]
        self.assertEqual(allowed_pattern, allowed_constraint.pattern)
        self.assertEqual(constraint_desc, allowed_constraint.description)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_string_multi_constraints(self):
        description = 'WebServer EC2 instance type'
        allowed_pattern = '[A-Za-z0-9.]*'
        constraint_desc = 'Must contain only alphanumeric characters.'
        param = parameters.Schema.from_dict('name', {'Type': 'String', 'Description': description, 'Default': 'm1.large', 'MinLength': '7', 'AllowedPattern': allowed_pattern, 'ConstraintDescription': constraint_desc})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.STRING, schema.type)
        self.assertEqual(description, schema.description)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertEqual(2, len(schema.constraints))
        len_constraint = schema.constraints[0]
        allowed_constraint = schema.constraints[1]
        self.assertEqual(7, len_constraint.min)
        self.assertIsNone(len_constraint.max)
        self.assertEqual(allowed_pattern, allowed_constraint.pattern)
        self.assertEqual(constraint_desc, allowed_constraint.description)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_param_string_min_len(self):
        param = parameters.Schema.from_dict('name', {'Description': 'WebServer EC2 instance type', 'Type': 'String', 'Default': 'm1.large', 'MinLength': '7'})
        schema = properties.Schema.from_parameter(param)
        self.assertFalse(schema.required)
        self.assertIsNone(schema.default)
        self.assertEqual(1, len(schema.constraints))
        len_constraint = schema.constraints[0]
        self.assertEqual(7, len_constraint.min)
        self.assertIsNone(len_constraint.max)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_param_string_max_len(self):
        param = parameters.Schema.from_dict('name', {'Description': 'WebServer EC2 instance type', 'Type': 'String', 'Default': 'm1.large', 'MaxLength': '11'})
        schema = properties.Schema.from_parameter(param)
        self.assertFalse(schema.required)
        self.assertIsNone(schema.default)
        self.assertEqual(1, len(schema.constraints))
        len_constraint = schema.constraints[0]
        self.assertIsNone(len_constraint.min)
        self.assertEqual(11, len_constraint.max)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_param_string_min_max_len(self):
        param = parameters.Schema.from_dict('name', {'Description': 'WebServer EC2 instance type', 'Type': 'String', 'Default': 'm1.large', 'MinLength': '7', 'MaxLength': '11'})
        schema = properties.Schema.from_parameter(param)
        self.assertFalse(schema.required)
        self.assertIsNone(schema.default)
        self.assertEqual(1, len(schema.constraints))
        len_constraint = schema.constraints[0]
        self.assertEqual(7, len_constraint.min)
        self.assertEqual(11, len_constraint.max)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_param_no_default(self):
        param = parameters.Schema.from_dict('name', {'Description': 'WebServer EC2 instance type', 'Type': 'String'})
        schema = properties.Schema.from_parameter(param)
        self.assertTrue(schema.required)
        self.assertIsNone(schema.default)
        self.assertEqual(0, len(schema.constraints))
        self.assertFalse(schema.allow_conversion)
        props = properties.Properties({'name': schema}, {'name': 'm1.large'})
        props.validate()

    def test_from_number_param_min(self):
        param = parameters.Schema.from_dict('name', {'Type': 'Number', 'Default': '42', 'MinValue': '10'})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.NUMBER, schema.type)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertEqual(1, len(schema.constraints))
        value_constraint = schema.constraints[0]
        self.assertEqual(10, value_constraint.min)
        self.assertIsNone(value_constraint.max)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_number_param_max(self):
        param = parameters.Schema.from_dict('name', {'Type': 'Number', 'Default': '42', 'MaxValue': '100'})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.NUMBER, schema.type)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertEqual(1, len(schema.constraints))
        value_constraint = schema.constraints[0]
        self.assertIsNone(value_constraint.min)
        self.assertEqual(100, value_constraint.max)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_number_param_min_max(self):
        param = parameters.Schema.from_dict('name', {'Type': 'Number', 'Default': '42', 'MinValue': '10', 'MaxValue': '100'})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.NUMBER, schema.type)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertEqual(1, len(schema.constraints))
        value_constraint = schema.constraints[0]
        self.assertEqual(10, value_constraint.min)
        self.assertEqual(100, value_constraint.max)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_number_param_allowed_vals(self):
        constraint_desc = 'The quick brown fox jumps over the lazy dog.'
        param = parameters.Schema.from_dict('name', {'Type': 'Number', 'Default': '42', 'AllowedValues': ['10', '42', '100'], 'ConstraintDescription': constraint_desc})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.NUMBER, schema.type)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertEqual(1, len(schema.constraints))
        self.assertFalse(schema.allow_conversion)
        allowed_constraint = schema.constraints[0]
        self.assertEqual(('10', '42', '100'), allowed_constraint.allowed)
        self.assertEqual(constraint_desc, allowed_constraint.description)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_list_param(self):
        param = parameters.Schema.from_dict('name', {'Type': 'CommaDelimitedList', 'Default': 'foo,bar,baz'})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.LIST, schema.type)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertTrue(schema.allow_conversion)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_from_json_param(self):
        param = parameters.Schema.from_dict('name', {'Type': 'Json', 'Default': {'foo': 'bar', 'blarg': 'wibble'}})
        schema = properties.Schema.from_parameter(param)
        self.assertEqual(properties.Schema.MAP, schema.type)
        self.assertIsNone(schema.default)
        self.assertFalse(schema.required)
        self.assertTrue(schema.allow_conversion)
        props = properties.Properties({'test': schema}, {})
        props.validate()

    def test_no_mismatch_in_update_policy(self):
        manager = plugin_manager.PluginManager('heat.engine.resources')
        resource_mapping = plugin_manager.PluginMapping('resource')
        res_plugin_mappings = resource_mapping.load_all(manager)
        all_resources = {}
        for mapping in res_plugin_mappings:
            name, cls = mapping
            all_resources[name] = cls

        def check_update_policy(resource_type, prop_key, prop, update=False):
            if prop.update_allowed:
                update = True
            sub_schema = prop.schema
            if sub_schema:
                for sub_prop_key, sub_prop in sub_schema.items():
                    if not update:
                        self.assertEqual(update, sub_prop.update_allowed, "Mismatch in update policies: resource %(res)s, properties '%(prop)s' and '%(nested_prop)s'." % {'res': resource_type, 'prop': prop_key, 'nested_prop': sub_prop_key})
                    if sub_prop_key == '*':
                        check_update_policy(resource_type, prop_key, sub_prop, update)
                    else:
                        check_update_policy(resource_type, sub_prop_key, sub_prop, update)
        for resource_type, resource_class in all_resources.items():
            props_schemata = properties.schemata(resource_class.properties_schema)
            for prop_key, prop in props_schemata.items():
                check_update_policy(resource_type, prop_key, prop)