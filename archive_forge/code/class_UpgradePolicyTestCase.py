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
class UpgradePolicyTestCase(base.PolicyBaseTestCase):

    def setUp(self):
        super(UpgradePolicyTestCase, self).setUp()
        policy_json_contents = jsonutils.dumps({'deprecated_name': 'rule:admin'})
        self.create_config_file('policy.json', policy_json_contents)
        deprecated_policy = policy.DeprecatedRule(name='deprecated_name', check_str='rule:admin', deprecated_reason='test', deprecated_since='Stein')
        self.new_policy = policy.DocumentedRuleDefault(name='new_policy_name', check_str='rule:admin', description='test_policy', operations=[{'path': '/test', 'method': 'GET'}], deprecated_rule=deprecated_policy)
        self.extensions = []
        ext = stevedore.extension.Extension(name='test_upgrade', entry_point=None, plugin=None, obj=[self.new_policy])
        self.extensions.append(ext)
        self.local_conf = cfg.ConfigOpts()

    def test_upgrade_policy_json_file(self):
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test_upgrade')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
            testargs = ['olsopolicy-policy-upgrade', '--policy', self.get_config_file_fullname('policy.json'), '--namespace', 'test_upgrade', '--output-file', self.get_config_file_fullname('new_policy.json'), '--format', 'json']
            with mock.patch('sys.argv', testargs):
                generator.upgrade_policy(conf=self.local_conf)
                new_file = self.get_config_file_fullname('new_policy.json')
                with open(new_file, 'r') as fh:
                    new_policy = jsonutils.loads(fh.read())
                self.assertIsNotNone(new_policy.get('new_policy_name'))
                self.assertIsNone(new_policy.get('deprecated_name'))

    @mock.patch.object(generator, 'LOG')
    def test_upgrade_policy_json_file_log_warning(self, mock_log):
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test_upgrade')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
            testargs = ['olsopolicy-policy-upgrade', '--policy', self.get_config_file_fullname('policy.json'), '--namespace', 'test_upgrade', '--output-file', self.get_config_file_fullname('new_policy.json'), '--format', 'json']
            with mock.patch('sys.argv', testargs):
                generator.upgrade_policy(conf=self.local_conf)
                mock_log.warning.assert_any_call(policy.WARN_JSON)

    def test_upgrade_policy_yaml_file(self):
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test_upgrade')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
            testargs = ['olsopolicy-policy-upgrade', '--policy', self.get_config_file_fullname('policy.json'), '--namespace', 'test_upgrade', '--output-file', self.get_config_file_fullname('new_policy.yaml'), '--format', 'yaml']
            with mock.patch('sys.argv', testargs):
                generator.upgrade_policy(conf=self.local_conf)
                new_file = self.get_config_file_fullname('new_policy.yaml')
                with open(new_file, 'r') as fh:
                    new_policy = yaml.safe_load(fh)
                self.assertIsNotNone(new_policy.get('new_policy_name'))
                self.assertIsNone(new_policy.get('deprecated_name'))

    def test_upgrade_policy_json_stdout(self):
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test_upgrade')
        stdout = self._capture_stdout()
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
            testargs = ['olsopolicy-policy-upgrade', '--policy', self.get_config_file_fullname('policy.json'), '--namespace', 'test_upgrade', '--format', 'json']
            with mock.patch('sys.argv', testargs):
                generator.upgrade_policy(conf=self.local_conf)
                expected = '{\n    "new_policy_name": "rule:admin"\n}'
                self.assertEqual(expected, stdout.getvalue())

    def test_upgrade_policy_yaml_stdout(self):
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=self.extensions, namespace='test_upgrade')
        stdout = self._capture_stdout()
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr):
            testargs = ['olsopolicy-policy-upgrade', '--policy', self.get_config_file_fullname('policy.json'), '--namespace', 'test_upgrade', '--format', 'yaml']
            with mock.patch('sys.argv', testargs):
                generator.upgrade_policy(conf=self.local_conf)
                expected = 'new_policy_name: rule:admin\n'
                self.assertEqual(expected, stdout.getvalue())