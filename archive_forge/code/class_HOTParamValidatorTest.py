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
class HOTParamValidatorTest(common.HeatTestCase):
    """Test HOTParamValidator."""

    def test_multiple_constraint_descriptions(self):
        len_desc = 'string length should be between 8 and 16'
        pattern_desc1 = 'Value must consist of characters only'
        pattern_desc2 = 'Value must start with a lowercase character'
        param = {'db_name': {'description': 'The WordPress database name', 'type': 'string', 'default': 'wordpress', 'constraints': [{'length': {'min': 6, 'max': 16}, 'description': len_desc}, {'allowed_pattern': '[a-zA-Z]+', 'description': pattern_desc1}, {'allowed_pattern': '[a-z]+[a-zA-Z]*', 'description': pattern_desc2}]}}
        name = 'db_name'
        schema = param['db_name']

        def v(value):
            param_schema = hot_param.HOTParamSchema.from_dict(name, schema)
            param_schema.validate()
            param_schema.validate_value(value)
            return True
        value = 'wp'
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(len_desc, str(err))
        value = 'abcdefghijklmnopq'
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(len_desc, str(err))
        value = 'abcdefgh1'
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(pattern_desc1, str(err))
        value = 'Abcdefghi'
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(pattern_desc2, str(err))
        value = 'abcdefghi'
        self.assertTrue(v(value))
        value = 'abcdefghI'
        self.assertTrue(v(value))

    def test_hot_template_validate_param(self):
        len_desc = 'string length should be between 8 and 16'
        pattern_desc1 = 'Value must consist of characters only'
        pattern_desc2 = 'Value must start with a lowercase character'
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n          db_name:\n            description: The WordPress database name\n            type: string\n            default: wordpress\n            constraints:\n              - length: { min: 8, max: 16 }\n                description: %s\n              - allowed_pattern: "[a-zA-Z]+"\n                description: %s\n              - allowed_pattern: "[a-z]+[a-zA-Z]*"\n                description: %s\n        ' % (len_desc, pattern_desc1, pattern_desc2))
        tmpl = template.Template(hot_tpl)

        def run_parameters(value):
            tmpl.parameters(identifier.HeatIdentifier('', 'stack_testit', None), {'db_name': value}).validate(validate_value=True)
            return True
        value = 'wp'
        err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
        self.assertIn(len_desc, str(err))
        value = 'abcdefghijklmnopq'
        err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
        self.assertIn(len_desc, str(err))
        value = 'abcdefgh1'
        err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
        self.assertIn(pattern_desc1, str(err))
        value = 'Abcdefghi'
        err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
        self.assertIn(pattern_desc2, str(err))
        value = 'abcdefghi'
        self.assertTrue(run_parameters(value))
        value = 'abcdefghI'
        self.assertTrue(run_parameters(value))

    def test_range_constraint(self):
        range_desc = 'Value must be between 30000 and 50000'
        param = {'db_port': {'description': 'The database port', 'type': 'number', 'default': 31000, 'constraints': [{'range': {'min': 30000, 'max': 50000}, 'description': range_desc}]}}
        name = 'db_port'
        schema = param['db_port']

        def v(value):
            param_schema = hot_param.HOTParamSchema.from_dict(name, schema)
            param_schema.validate()
            param_schema.validate_value(value)
            return True
        value = 29999
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(range_desc, str(err))
        value = 50001
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(range_desc, str(err))
        value = 30000
        self.assertTrue(v(value))
        value = 40000
        self.assertTrue(v(value))
        value = 50000
        self.assertTrue(v(value))

    def test_custom_constraint(self):

        class ZeroConstraint(object):

            def validate(self, value, context):
                return value == '0'
        env = resources.global_env()
        env.register_constraint('zero', ZeroConstraint)
        self.addCleanup(env.constraints.pop, 'zero')
        desc = 'Value must be zero'
        param = {'param1': {'type': 'string', 'constraints': [{'custom_constraint': 'zero', 'description': desc}]}}
        name = 'param1'
        schema = param['param1']

        def v(value):
            param_schema = hot_param.HOTParamSchema.from_dict(name, schema)
            param_schema.validate()
            param_schema.validate_value(value)
            return True
        value = '1'
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertEqual(desc, str(err))
        value = '2'
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertEqual(desc, str(err))
        value = '0'
        self.assertTrue(v(value))

    def test_custom_constraint_default_skip(self):
        schema = {'type': 'string', 'constraints': [{'custom_constraint': 'skipping', 'description': 'Must be skipped on default value'}], 'default': 'foo'}
        param_schema = hot_param.HOTParamSchema.from_dict('p', schema)
        param_schema.validate()

    def test_range_constraint_invalid_default(self):
        range_desc = 'Value must be between 30000 and 50000'
        param = {'db_port': {'description': 'The database port', 'type': 'number', 'default': 15, 'constraints': [{'range': {'min': 30000, 'max': 50000}, 'description': range_desc}]}}
        schema = hot_param.HOTParamSchema.from_dict('db_port', param['db_port'])
        err = self.assertRaises(exception.InvalidSchemaError, schema.validate)
        self.assertIn(range_desc, str(err))

    def test_validate_schema_wrong_key(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                foo: bar\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual("Invalid key 'foo' for parameter (param1)", str(error))

    def test_validate_schema_no_type(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                description: Hi!\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('Missing parameter type for parameter: param1', str(error))

    def test_validate_schema_unknown_type(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: Unicode\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('Invalid type (Unicode)', str(error))

    def test_validate_schema_constraints(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                constraints:\n                   - allowed_valus: [foo, bar]\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual("Invalid key 'allowed_valus' for parameter constraints", str(error))

    def test_validate_schema_constraints_not_list(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                constraints: 1\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('Invalid parameter constraints for parameter param1, expected a list', str(error))

    def test_validate_schema_constraints_not_mapping(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                constraints: [foo]\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('Invalid parameter constraints, expected a mapping', str(error))

    def test_validate_schema_empty_constraints(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                constraints:\n                    - description: a constraint\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('No constraint expressed', str(error))

    def test_validate_schema_constraints_range_wrong_format(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: number\n                constraints:\n                   - range: foo\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('Invalid range constraint, expected a mapping', str(error))

    def test_validate_schema_constraints_range_invalid_key(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: number\n                constraints:\n                    - range: {min: 1, foo: bar}\n                default: 1\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual("Invalid key 'foo' for range constraint", str(error))

    def test_validate_schema_constraints_length_wrong_format(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                constraints:\n                   - length: foo\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('Invalid length constraint, expected a mapping', str(error))

    def test_validate_schema_constraints_length_invalid_key(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                constraints:\n                    - length: {min: 1, foo: bar}\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual("Invalid key 'foo' for length constraint", str(error))

    def test_validate_schema_constraints_wrong_allowed_pattern(self):
        hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n            param1:\n                type: string\n                constraints:\n                    - allowed_pattern: [foo, bar]\n                default: foo\n        ')
        error = self.assertRaises(exception.InvalidSchemaError, cfn_param.CfnParameters, 'stack_testit', template.Template(hot_tpl))
        self.assertEqual('AllowedPattern must be a string', str(error))

    def test_modulo_constraint(self):
        modulo_desc = 'Value must be an odd number'
        modulo_name = 'ControllerCount'
        param = {modulo_name: {'description': 'Number of controller nodes', 'type': 'number', 'default': 1, 'constraints': [{'modulo': {'step': 2, 'offset': 1}, 'description': modulo_desc}]}}

        def v(value):
            param_schema = hot_param.HOTParamSchema20170224.from_dict(modulo_name, param[modulo_name])
            param_schema.validate()
            param_schema.validate_value(value)
            return True
        value = 2
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(modulo_desc, str(err))
        value = 100
        err = self.assertRaises(exception.StackValidationFailed, v, value)
        self.assertIn(modulo_desc, str(err))
        value = 1
        self.assertTrue(v(value))
        value = 3
        self.assertTrue(v(value))
        value = 777
        self.assertTrue(v(value))

    def test_modulo_constraint_invalid_default(self):
        modulo_desc = 'Value must be an odd number'
        modulo_name = 'ControllerCount'
        param = {modulo_name: {'description': 'Number of controller nodes', 'type': 'number', 'default': 2, 'constraints': [{'modulo': {'step': 2, 'offset': 1}, 'description': modulo_desc}]}}
        schema = hot_param.HOTParamSchema20170224.from_dict(modulo_name, param[modulo_name])
        err = self.assertRaises(exception.InvalidSchemaError, schema.validate)
        self.assertIn(modulo_desc, str(err))