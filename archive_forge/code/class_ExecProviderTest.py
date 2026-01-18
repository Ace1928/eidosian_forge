import os
import unittest
import mock
from .config_exception import ConfigException
from .exec_provider import ExecProvider
from .kube_config import ConfigNode
class ExecProviderTest(unittest.TestCase):

    def setUp(self):
        self.input_ok = ConfigNode('test', {'command': 'aws-iam-authenticator', 'args': ['token', '-i', 'dummy'], 'apiVersion': 'client.authentication.k8s.io/v1beta1', 'env': None})
        self.output_ok = '\n        {\n            "apiVersion": "client.authentication.k8s.io/v1beta1",\n            "kind": "ExecCredential",\n            "status": {\n                "token": "dummy"\n            }\n        }\n        '

    def test_missing_input_keys(self):
        exec_configs = [ConfigNode('test1', {}), ConfigNode('test2', {'command': ''}), ConfigNode('test3', {'apiVersion': ''})]
        for exec_config in exec_configs:
            with self.assertRaises(ConfigException) as context:
                ExecProvider(exec_config)
            self.assertIn('exec: malformed request. missing key', context.exception.args[0])

    @mock.patch('subprocess.Popen')
    def test_error_code_returned(self, mock):
        instance = mock.return_value
        instance.wait.return_value = 1
        instance.communicate.return_value = ('', '')
        with self.assertRaises(ConfigException) as context:
            ep = ExecProvider(self.input_ok)
            ep.run()
        self.assertIn('exec: process returned %d' % instance.wait.return_value, context.exception.args[0])

    @mock.patch('subprocess.Popen')
    def test_nonjson_output_returned(self, mock):
        instance = mock.return_value
        instance.wait.return_value = 0
        instance.communicate.return_value = ('', '')
        with self.assertRaises(ConfigException) as context:
            ep = ExecProvider(self.input_ok)
            ep.run()
        self.assertIn('exec: failed to decode process output', context.exception.args[0])

    @mock.patch('subprocess.Popen')
    def test_missing_output_keys(self, mock):
        instance = mock.return_value
        instance.wait.return_value = 0
        outputs = ['\n            {\n                "kind": "ExecCredential",\n                "status": {\n                    "token": "dummy"\n                }\n            }\n            ', '\n            {\n                "apiVersion": "client.authentication.k8s.io/v1beta1",\n                "status": {\n                    "token": "dummy"\n                }\n            }\n            ', '\n            {\n                "apiVersion": "client.authentication.k8s.io/v1beta1",\n                "kind": "ExecCredential"\n            }\n            ']
        for output in outputs:
            instance.communicate.return_value = (output, '')
            with self.assertRaises(ConfigException) as context:
                ep = ExecProvider(self.input_ok)
                ep.run()
            self.assertIn('exec: malformed response. missing key', context.exception.args[0])

    @mock.patch('subprocess.Popen')
    def test_mismatched_api_version(self, mock):
        instance = mock.return_value
        instance.wait.return_value = 0
        wrong_api_version = 'client.authentication.k8s.io/v1'
        output = '\n        {\n            "apiVersion": "%s",\n            "kind": "ExecCredential",\n            "status": {\n                "token": "dummy"\n            }\n        }\n        ' % wrong_api_version
        instance.communicate.return_value = (output, '')
        with self.assertRaises(ConfigException) as context:
            ep = ExecProvider(self.input_ok)
            ep.run()
        self.assertIn('exec: plugin api version %s does not match' % wrong_api_version, context.exception.args[0])

    @mock.patch('subprocess.Popen')
    def test_ok_01(self, mock):
        instance = mock.return_value
        instance.wait.return_value = 0
        instance.communicate.return_value = (self.output_ok, '')
        ep = ExecProvider(self.input_ok)
        result = ep.run()
        self.assertTrue(isinstance(result, dict))
        self.assertTrue('token' in result)