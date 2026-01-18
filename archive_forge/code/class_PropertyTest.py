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
class PropertyTest(common.HeatTestCase):

    def test_required_default(self):
        p = properties.Property({'Type': 'String'})
        self.assertFalse(p.required())

    def test_required_false(self):
        p = properties.Property({'Type': 'String', 'Required': False})
        self.assertFalse(p.required())

    def test_required_true(self):
        p = properties.Property({'Type': 'String', 'Required': True})
        self.assertTrue(p.required())

    def test_implemented_default(self):
        p = properties.Property({'Type': 'String'})
        self.assertTrue(p.implemented())

    def test_implemented_false(self):
        p = properties.Property({'Type': 'String', 'Implemented': False})
        self.assertFalse(p.implemented())

    def test_implemented_true(self):
        p = properties.Property({'Type': 'String', 'Implemented': True})
        self.assertTrue(p.implemented())

    def test_no_default(self):
        p = properties.Property({'Type': 'String'})
        self.assertFalse(p.has_default())

    def test_default(self):
        p = properties.Property({'Type': 'String', 'Default': 'wibble'})
        self.assertEqual('wibble', p.default())

    def test_type(self):
        p = properties.Property({'Type': 'String'})
        self.assertEqual('String', p.type())

    def test_bad_type(self):
        self.assertRaises(exception.InvalidSchemaError, properties.Property, {'Type': 'Fish'})

    def test_bad_key(self):
        self.assertRaises(exception.InvalidSchemaError, properties.Property, {'Type': 'String', 'Foo': 'Bar'})

    def test_string_pattern_good(self):
        schema = {'Type': 'String', 'AllowedPattern': '[a-z]*'}
        p = properties.Property(schema)
        self.assertEqual('foo', p.get_value('foo', True))

    def test_string_pattern_bad_prefix(self):
        schema = {'Type': 'String', 'AllowedPattern': '[a-z]*'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, '1foo', True)

    def test_string_pattern_bad_suffix(self):
        schema = {'Type': 'String', 'AllowedPattern': '[a-z]*'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, 'foo1', True)

    def test_string_value_list_good(self):
        schema = {'Type': 'String', 'AllowedValues': ['foo', 'bar', 'baz']}
        p = properties.Property(schema)
        self.assertEqual('bar', p.get_value('bar', True))

    def test_string_value_list_bad(self):
        schema = {'Type': 'String', 'AllowedValues': ['foo', 'bar', 'baz']}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, 'blarg', True)

    def test_string_maxlength_good(self):
        schema = {'Type': 'String', 'MaxLength': '5'}
        p = properties.Property(schema)
        self.assertEqual('abcd', p.get_value('abcd', True))

    def test_string_exceeded_maxlength(self):
        schema = {'Type': 'String', 'MaxLength': '5'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, 'abcdef', True)

    def test_string_length_in_range(self):
        schema = {'Type': 'String', 'MinLength': '5', 'MaxLength': '10'}
        p = properties.Property(schema)
        self.assertEqual('abcdef', p.get_value('abcdef', True))

    def test_string_minlength_good(self):
        schema = {'Type': 'String', 'MinLength': '5'}
        p = properties.Property(schema)
        self.assertEqual('abcde', p.get_value('abcde', True))

    def test_string_smaller_than_minlength(self):
        schema = {'Type': 'String', 'MinLength': '5'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, 'abcd', True)

    def test_int_good(self):
        schema = {'Type': 'Integer', 'MinValue': 3, 'MaxValue': 3}
        p = properties.Property(schema)
        self.assertEqual(3, p.get_value(3, True))

    def test_int_bad(self):
        schema = {'Type': 'Integer'}
        p = properties.Property(schema)
        self.assertRaisesRegex(TypeError, "int\\(\\) argument must be a string(, a bytes-like object)? or a (real )?number, not 'list'", p.get_value, [1])

    def test_str_from_int(self):
        schema = {'Type': 'String'}
        p = properties.Property(schema)
        self.assertEqual('3', p.get_value(3))

    def test_str_from_bool(self):
        schema = {'Type': 'String'}
        p = properties.Property(schema)
        self.assertEqual('True', p.get_value(True))

    def test_int_from_str_good(self):
        schema = {'Type': 'Integer'}
        p = properties.Property(schema)
        self.assertEqual(3, p.get_value('3'))

    def test_int_from_str_bad(self):
        schema = {'Type': 'Integer'}
        p = properties.Property(schema)
        ex = self.assertRaises(TypeError, p.get_value, '3a')
        self.assertEqual("Value '3a' is not an integer", str(ex))

    def test_integer_low(self):
        schema = {'Type': 'Integer', 'MinValue': 4}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, 3, True)

    def test_integer_high(self):
        schema = {'Type': 'Integer', 'MaxValue': 2}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, 3, True)

    def test_integer_value_list_good(self):
        schema = {'Type': 'Integer', 'AllowedValues': [1, 3, 5]}
        p = properties.Property(schema)
        self.assertEqual(5, p.get_value(5), True)

    def test_integer_value_list_bad(self):
        schema = {'Type': 'Integer', 'AllowedValues': [1, 3, 5]}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, 2, True)

    def test_number_good(self):
        schema = {'Type': 'Number', 'MinValue': '3', 'MaxValue': '3'}
        p = properties.Property(schema)
        self.assertEqual(3, p.get_value(3, True))

    def test_numbers_from_strings(self):
        """Numbers can be converted from strings."""
        schema = {'Type': 'Number', 'MinValue': '3', 'MaxValue': '3'}
        p = properties.Property(schema)
        self.assertEqual(3, p.get_value('3'))

    def test_number_value_list_good(self):
        schema = {'Type': 'Number', 'AllowedValues': [1, 3, 5]}
        p = properties.Property(schema)
        self.assertEqual(5, p.get_value('5', True))

    def test_number_value_list_bad(self):
        schema = {'Type': 'Number', 'AllowedValues': ['1', '3', '5']}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, '2', True)

    def test_number_low(self):
        schema = {'Type': 'Number', 'MinValue': '4'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, '3', True)

    def test_number_high(self):
        schema = {'Type': 'Number', 'MaxValue': '2'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, '3', True)

    def test_boolean_true(self):
        p = properties.Property({'Type': 'Boolean'})
        self.assertIs(True, p.get_value('True'))
        self.assertIs(True, p.get_value('true'))
        self.assertIs(True, p.get_value(True))

    def test_boolean_false(self):
        p = properties.Property({'Type': 'Boolean'})
        self.assertIs(False, p.get_value('False'))
        self.assertIs(False, p.get_value('false'))
        self.assertIs(False, p.get_value(False))

    def test_boolean_invalid_string(self):
        p = properties.Property({'Type': 'Boolean'})
        self.assertRaises(ValueError, p.get_value, 'fish')

    def test_boolean_invalid_int(self):
        p = properties.Property({'Type': 'Boolean'})
        self.assertRaises(TypeError, p.get_value, 5)

    def test_list_string(self):
        p = properties.Property({'Type': 'List'})
        self.assertRaises(TypeError, p.get_value, 'foo')

    def test_list_good(self):
        p = properties.Property({'Type': 'List'})
        self.assertEqual(['foo', 'bar'], p.get_value(['foo', 'bar']))

    def test_list_dict(self):
        p = properties.Property({'Type': 'List'})
        self.assertRaises(TypeError, p.get_value, {'foo': 'bar'})

    def test_list_is_delimited(self):
        p = properties.Property({'Type': 'List'})
        self.assertRaises(TypeError, p.get_value, 'foo,bar')
        p.schema.allow_conversion = True
        self.assertEqual(['foo', 'bar'], p.get_value('foo,bar'))
        self.assertEqual(['foo'], p.get_value('foo'))

    def test_map_path(self):
        p = properties.Property({'Type': 'Map'}, name='test', path='parent')
        self.assertEqual('parent.test', p.path)

    def test_list_path(self):
        p = properties.Property({'Type': 'List'}, name='0', path='parent')
        self.assertEqual('parent.0', p.path)

    def test_list_maxlength_good(self):
        schema = {'Type': 'List', 'MaxLength': '3'}
        p = properties.Property(schema)
        self.assertEqual(['1', '2'], p.get_value(['1', '2'], True))

    def test_list_exceeded_maxlength(self):
        schema = {'Type': 'List', 'MaxLength': '2'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, ['1', '2', '3'], True)

    def test_list_length_in_range(self):
        schema = {'Type': 'List', 'MinLength': '2', 'MaxLength': '4'}
        p = properties.Property(schema)
        self.assertEqual(['1', '2', '3'], p.get_value(['1', '2', '3'], True))

    def test_list_minlength_good(self):
        schema = {'Type': 'List', 'MinLength': '3'}
        p = properties.Property(schema)
        self.assertEqual(['1', '2', '3'], p.get_value(['1', '2', '3'], True))

    def test_list_smaller_than_minlength(self):
        schema = {'Type': 'List', 'MinLength': '4'}
        p = properties.Property(schema)
        self.assertRaises(exception.StackValidationFailed, p.get_value, ['1', '2', '3'], True)

    def test_map_list_default(self):
        schema = {'Type': 'Map', 'Default': ['foo', 'bar']}
        p = properties.Property(schema)
        p.schema.allow_conversion = True
        self.assertEqual(jsonutils.dumps(['foo', 'bar']), p.get_value(None))

    def test_map_list_default_empty(self):
        schema = {'Type': 'Map', 'Default': []}
        p = properties.Property(schema)
        p.schema.allow_conversion = True
        self.assertEqual(jsonutils.dumps([]), p.get_value(None))

    def test_map_list_no_default(self):
        schema = {'Type': 'Map'}
        p = properties.Property(schema)
        p.schema.allow_conversion = True
        self.assertEqual({}, p.get_value(None))

    def test_map_string(self):
        p = properties.Property({'Type': 'Map'})
        self.assertRaises(TypeError, p.get_value, 'foo')

    def test_map_list(self):
        p = properties.Property({'Type': 'Map'})
        self.assertRaises(TypeError, p.get_value, ['foo'])

    def test_map_allow_conversion(self):
        p = properties.Property({'Type': 'Map'})
        p.schema.allow_conversion = True
        self.assertEqual('foo', p.get_value('foo'))
        self.assertEqual(jsonutils.dumps(['foo']), p.get_value(['foo']))

    def test_map_schema_good(self):
        map_schema = {'valid': {'Type': 'Boolean'}}
        p = properties.Property({'Type': 'Map', 'Schema': map_schema})
        self.assertEqual({'valid': True}, p.get_value({'valid': 'TRUE'}))

    def test_map_schema_bad_data(self):
        map_schema = {'valid': {'Type': 'Boolean'}}
        p = properties.Property({'Type': 'Map', 'Schema': map_schema})
        ex = self.assertRaises(exception.StackValidationFailed, p.get_value, {'valid': 'fish'}, True)
        self.assertEqual('Property error: valid: "fish" is not a valid boolean', str(ex))

    def test_map_schema_missing_data(self):
        map_schema = {'valid': {'Type': 'Boolean'}}
        p = properties.Property({'Type': 'Map', 'Schema': map_schema})
        self.assertEqual({'valid': None}, p.get_value({}))

    def test_map_schema_missing_required_data(self):
        map_schema = {'valid': {'Type': 'Boolean', 'Required': True}}
        p = properties.Property({'Type': 'Map', 'Schema': map_schema})
        ex = self.assertRaises(exception.StackValidationFailed, p.get_value, {}, True)
        self.assertEqual('Property error: Property valid not assigned', str(ex))

    def test_list_schema_good(self):
        map_schema = {'valid': {'Type': 'Boolean'}}
        list_schema = {'Type': 'Map', 'Schema': map_schema}
        p = properties.Property({'Type': 'List', 'Schema': list_schema})
        self.assertEqual([{'valid': True}, {'valid': False}], p.get_value([{'valid': 'TRUE'}, {'valid': 'False'}]))

    def test_list_schema_bad_data(self):
        map_schema = {'valid': {'Type': 'Boolean'}}
        list_schema = {'Type': 'Map', 'Schema': map_schema}
        p = properties.Property({'Type': 'List', 'Schema': list_schema})
        ex = self.assertRaises(exception.StackValidationFailed, p.get_value, [{'valid': 'True'}, {'valid': 'fish'}], True)
        self.assertEqual('Property error: [1].valid: "fish" is not a valid boolean', str(ex))

    def test_list_schema_int_good(self):
        list_schema = {'Type': 'Integer'}
        p = properties.Property({'Type': 'List', 'Schema': list_schema})
        self.assertEqual([1, 2, 3], p.get_value([1, 2, 3]))

    def test_list_schema_int_bad_data(self):
        list_schema = {'Type': 'Integer'}
        p = properties.Property({'Type': 'List', 'Schema': list_schema})
        ex = self.assertRaises(exception.StackValidationFailed, p.get_value, [42, 'fish'], True)
        self.assertEqual("Property error: [1]: Value 'fish' is not an integer", str(ex))