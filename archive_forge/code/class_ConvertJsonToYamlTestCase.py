import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
class ConvertJsonToYamlTestCase(base.PolicyBaseTestCase):

    def setUp(self):
        super(ConvertJsonToYamlTestCase, self).setUp()
        policy_json_contents = jsonutils.dumps({'rule1_name': 'rule:admin', 'rule2_name': 'rule:overridden', 'deprecated_rule1_name': 'rule:admin'})
        self.create_config_file('policy.json', policy_json_contents)
        self.output_file_path = self.get_config_file_fullname('converted_policy.yaml')
        deprecated_policy = policy.DeprecatedRule(name='deprecated_rule1_name', check_str='rule:admin', deprecated_reason='testing', deprecated_since='ussuri')
        self.registered_policy = [policy.DocumentedRuleDefault(name='rule1_name', check_str='rule:admin', description='test_rule1', operations=[{'path': '/test', 'method': 'GET'}], deprecated_rule=deprecated_policy, scope_types=['system']), policy.RuleDefault(name='rule2_name', check_str='rule:admin')]
        self.extensions = []
        ext = stevedore.extension.Extension(name='test', entry_point=None, plugin=None, obj=self.registered_policy)
        self.extensions.append(ext)
        self.local_conf = cfg.ConfigOpts()
        self.expected = '# test_rule1\n# GET  /test\n# Intended scope(s): system\n#"rule1_name": "rule:admin"\n\n# rule2_name\n"rule2_name": "rule:overridden"\n\n# WARNING: Below rules are either deprecated rules\n# or extra rules in policy file, it is strongly\n# recommended to switch to new rules.\n"deprecated_rule1_name": "rule:admin"\n'

    def _is_yaml(self, data):
        is_yaml = False
        try:
            jsonutils.loads(data)
        except ValueError:
            try:
                yaml.safe_load(data)
                is_yaml = True
            except yaml.scanner.ScannerError:
                pass
        return is_yaml

    def _test_convert_json_to_yaml_file(self, output_to_file=True):
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test')
        converted_policy_data = None
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
            testargs = ['oslopolicy-convert-json-to-yaml', '--namespace', 'test', '--policy-file', self.get_config_file_fullname('policy.json')]
            if output_to_file:
                testargs.extend(['--output-file', self.output_file_path])
            with mock.patch('sys.argv', testargs):
                generator.convert_policy_json_to_yaml(conf=self.local_conf)
                if output_to_file:
                    with open(self.output_file_path, 'r') as fh:
                        converted_policy_data = fh.read()
        return converted_policy_data

    def test_convert_json_to_yaml_file(self):
        converted_policy_data = self._test_convert_json_to_yaml_file()
        self.assertTrue(self._is_yaml(converted_policy_data))
        self.assertEqual(self.expected, converted_policy_data)

    def test_convert_policy_to_stdout(self):
        stdout = self._capture_stdout()
        self._test_convert_json_to_yaml_file(output_to_file=False)
        self.assertEqual(self.expected, stdout.getvalue())

    def test_converted_yaml_is_loadable(self):
        self._test_convert_json_to_yaml_file()
        enforcer = policy.Enforcer(self.conf, policy_file=self.output_file_path)
        enforcer.load_rules()
        for rule in ['rule2_name', 'deprecated_rule1_name']:
            self.assertIn(rule, enforcer.rules)

    def test_default_rules_comment_out_in_yaml_file(self):
        converted_policy_data = self._test_convert_json_to_yaml_file()
        commented_default_rule = '# test_rule1\n# GET  /test\n# Intended scope(s): system\n#"rule1_name": "rule:admin"\n\n'
        self.assertIn(commented_default_rule, converted_policy_data)

    def test_overridden_rules_uncommented_in_yaml_file(self):
        converted_policy_data = self._test_convert_json_to_yaml_file()
        uncommented_overridden_rule = '# rule2_name\n"rule2_name": "rule:overridden"\n\n'
        self.assertIn(uncommented_overridden_rule, converted_policy_data)

    def test_existing_deprecated_rules_kept_uncommented_in_yaml_file(self):
        converted_policy_data = self._test_convert_json_to_yaml_file()
        existing_deprecated_rule_with_warning = '# WARNING: Below rules are either deprecated rules\n# or extra rules in policy file, it is strongly\n# recommended to switch to new rules.\n"deprecated_rule1_name": "rule:admin"\n'
        self.assertIn(existing_deprecated_rule_with_warning, converted_policy_data)