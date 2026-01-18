from unittest import mock
import yaml
from heat.common import environment_format
from heat.tests import common
class YamlEnvironmentTest(common.HeatTestCase):

    def test_minimal_yaml(self):
        yaml1 = ''
        yaml2 = '\nparameters: {}\nencrypted_param_names: []\nparameter_defaults: {}\nevent_sinks: []\nresource_registry: {}\n'
        tpl1 = environment_format.parse(yaml1)
        environment_format.default_for_missing(tpl1)
        tpl2 = environment_format.parse(yaml2)
        self.assertEqual(tpl1, tpl2)

    def test_param_valid_strategy_section(self):
        yaml1 = ''
        yaml2 = '\nparameters: {}\nencrypted_param_names: []\nparameter_defaults: {}\nparameter_merge_strategies: {}\nevent_sinks: []\nresource_registry: {}\n'
        tpl1 = environment_format.parse(yaml1)
        environment_format.default_for_missing(tpl1)
        tpl2 = environment_format.parse(yaml2)
        self.assertNotEqual(tpl1, tpl2)

    def test_wrong_sections(self):
        env = '\nparameters: {}\nresource_regis: {}\n'
        self.assertRaises(ValueError, environment_format.parse, env)

    def test_bad_yaml(self):
        env = '\nparameters: }\n'
        self.assertRaises(ValueError, environment_format.parse, env)

    def test_yaml_none(self):
        self.assertEqual({}, environment_format.parse(None))

    def test_parse_string_environment(self):
        env = 'just string'
        expect = 'The environment is not a valid YAML mapping data type.'
        msg = self.assertRaises(ValueError, environment_format.parse, env)
        self.assertIn(expect, msg.args)

    def test_parse_document(self):
        env = '["foo" , "bar"]'
        expect = 'The environment is not a valid YAML mapping data type.'
        msg = self.assertRaises(ValueError, environment_format.parse, env)
        self.assertIn(expect, msg.args)