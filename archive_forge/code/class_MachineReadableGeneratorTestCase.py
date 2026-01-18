import io
import sys
import textwrap
from unittest import mock
import fixtures
from oslotest import base
import tempfile
import testscenarios
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_config import generator
from oslo_config import types
import yaml
class MachineReadableGeneratorTestCase(base.BaseTestCase):
    all_opts = GeneratorTestCase.opts
    all_groups = GeneratorTestCase.groups
    content_scenarios = [('single_namespace', dict(opts=[('test', [(None, [all_opts['foo']])])], expected={'deprecated_options': {}, 'generator_options': GENERATOR_OPTS, 'options': {'DEFAULT': {'driver_option': '', 'driver_opts': {}, 'dynamic_group_owner': '', 'help': '', 'standard_opts': ['foo'], 'opts': [{'advanced': False, 'choices': [], 'default': None, 'deprecated_for_removal': False, 'deprecated_opts': [], 'deprecated_reason': None, 'deprecated_since': None, 'dest': 'foo', 'help': 'foo option', 'max': None, 'metavar': None, 'min': None, 'mutable': False, 'name': 'foo', 'namespace': 'test', 'positional': False, 'required': False, 'sample_default': None, 'secret': False, 'short': None, 'type': 'string value'}]}}})), ('long_help', dict(opts=[('test', [(None, [all_opts['long_help']])])], expected={'deprecated_options': {}, 'generator_options': GENERATOR_OPTS, 'options': {'DEFAULT': {'driver_option': '', 'driver_opts': {}, 'dynamic_group_owner': '', 'help': '', 'standard_opts': ['long_help'], 'opts': [{'advanced': False, 'choices': [], 'default': None, 'deprecated_for_removal': False, 'deprecated_opts': [], 'deprecated_reason': None, 'deprecated_since': None, 'dest': 'long_help', 'help': all_opts['long_help'].help, 'max': None, 'metavar': None, 'min': None, 'mutable': False, 'name': 'long_help', 'namespace': 'test', 'positional': False, 'required': False, 'sample_default': None, 'secret': False, 'short': None, 'type': 'string value'}]}}})), ('long_help_pre', dict(opts=[('test', [(None, [all_opts['long_help_pre']])])], expected={'deprecated_options': {}, 'generator_options': GENERATOR_OPTS, 'options': {'DEFAULT': {'driver_option': '', 'driver_opts': {}, 'dynamic_group_owner': '', 'help': '', 'standard_opts': ['long_help_pre'], 'opts': [{'advanced': False, 'choices': [], 'default': None, 'deprecated_for_removal': False, 'deprecated_opts': [], 'deprecated_reason': None, 'deprecated_since': None, 'dest': 'long_help_pre', 'help': all_opts['long_help_pre'].help, 'max': None, 'metavar': None, 'min': None, 'mutable': False, 'name': 'long_help_pre', 'namespace': 'test', 'positional': False, 'required': False, 'sample_default': None, 'secret': False, 'short': None, 'type': 'string value'}]}}})), ('opt_with_DeprecatedOpt', dict(opts=[('test', [(None, [all_opts['opt_with_DeprecatedOpt']])])], expected={'deprecated_options': {'deprecated': [{'name': 'foo_bar', 'replacement_group': 'DEFAULT', 'replacement_name': 'foo-bar'}]}, 'generator_options': GENERATOR_OPTS, 'options': {'DEFAULT': {'driver_option': '', 'driver_opts': {}, 'dynamic_group_owner': '', 'help': '', 'standard_opts': ['foo-bar'], 'opts': [{'advanced': False, 'choices': [], 'default': None, 'deprecated_for_removal': False, 'deprecated_opts': [{'group': 'deprecated', 'name': 'foo_bar'}], 'deprecated_reason': None, 'deprecated_since': None, 'dest': 'foo_bar', 'help': all_opts['opt_with_DeprecatedOpt'].help, 'max': None, 'metavar': None, 'min': None, 'mutable': False, 'name': 'foo-bar', 'namespace': 'test', 'positional': False, 'required': False, 'sample_default': None, 'secret': False, 'short': None, 'type': 'boolean value'}]}}})), ('choices_opt', dict(opts=[('test', [(None, [all_opts['choices_opt']])])], expected={'deprecated_options': {}, 'generator_options': GENERATOR_OPTS, 'options': {'DEFAULT': {'driver_option': '', 'driver_opts': {}, 'dynamic_group_owner': '', 'help': '', 'standard_opts': ['choices_opt'], 'opts': [{'advanced': False, 'choices': [(None, None), ('', None), ('a', None), ('b', None), ('c', None)], 'default': 'a', 'deprecated_for_removal': False, 'deprecated_opts': [], 'deprecated_reason': None, 'deprecated_since': None, 'dest': 'choices_opt', 'help': all_opts['choices_opt'].help, 'max': None, 'metavar': None, 'min': None, 'mutable': False, 'name': 'choices_opt', 'namespace': 'test', 'positional': False, 'required': False, 'sample_default': None, 'secret': False, 'short': None, 'type': 'string value'}]}}})), ('int_opt', dict(opts=[('test', [(None, [all_opts['int_opt']])])], expected={'deprecated_options': {}, 'generator_options': GENERATOR_OPTS, 'options': {'DEFAULT': {'driver_option': '', 'driver_opts': {}, 'dynamic_group_owner': '', 'help': '', 'standard_opts': ['int_opt'], 'opts': [{'advanced': False, 'choices': [], 'default': 10, 'deprecated_for_removal': False, 'deprecated_opts': [], 'deprecated_reason': None, 'deprecated_since': None, 'dest': 'int_opt', 'help': all_opts['int_opt'].help, 'max': 20, 'metavar': None, 'min': 1, 'mutable': False, 'name': 'int_opt', 'namespace': 'test', 'positional': False, 'required': False, 'sample_default': None, 'secret': False, 'short': None, 'type': 'integer value'}]}}})), ('group_help', dict(opts=[('test', [(all_groups['group1'], [all_opts['foo']])])], expected={'deprecated_options': {}, 'generator_options': GENERATOR_OPTS, 'options': {'DEFAULT': {'help': '', 'standard_opts': [], 'opts': []}, 'group1': {'driver_option': '', 'driver_opts': {}, 'dynamic_group_owner': '', 'help': all_groups['group1'].help, 'standard_opts': ['foo'], 'opts': [{'advanced': False, 'choices': [], 'default': None, 'deprecated_for_removal': False, 'deprecated_opts': [], 'deprecated_reason': None, 'deprecated_since': None, 'dest': 'foo', 'help': all_opts['foo'].help, 'max': None, 'metavar': None, 'min': None, 'mutable': False, 'name': 'foo', 'namespace': 'test', 'positional': False, 'required': False, 'sample_default': None, 'secret': False, 'short': None, 'type': 'string value'}]}}}))]

    def setUp(self):
        super(MachineReadableGeneratorTestCase, self).setUp()
        self.conf = cfg.ConfigOpts()
        self.config_fixture = config_fixture.Config(self.conf)
        self.config = self.config_fixture.config
        self.useFixture(self.config_fixture)

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls.content_scenarios)

    @mock.patch.object(generator, '_get_raw_opts_loaders')
    def test_generate(self, raw_opts_loader):
        generator.register_cli_opts(self.conf)
        namespaces = [i[0] for i in self.opts]
        self.config(namespace=namespaces, format_='yaml')
        raw_opts_loader.return_value = [(ns, lambda opts=opts: opts) for ns, opts in self.opts]
        test_groups = generator._get_groups(generator._list_opts(self.conf.namespace))
        self.assertEqual(self.expected, generator._generate_machine_readable_data(test_groups, self.conf))