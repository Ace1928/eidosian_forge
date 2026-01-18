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
def _test_formatting(self, description, expected):
    rule = [policy.RuleDefault('admin', 'is_admin:True', description=description)]
    ext = stevedore.extension.Extension(name='check_rule', entry_point=None, plugin=None, obj=rule)
    test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=[ext], namespace=['check_rule'])
    output_file = self.get_config_file_fullname('policy.yaml')
    with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
        generator._generate_sample(['check_rule'], output_file)
        mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['check_rule'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
    with open(output_file, 'r') as written_file:
        written_policy = written_file.read()
    self.assertEqual(expected, written_policy)