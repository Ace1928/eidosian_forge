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