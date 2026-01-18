import boto
from boto.configservice.exceptions import NoSuchConfigurationRecorderException
from tests.compat import unittest
class TestConfigService(unittest.TestCase):

    def setUp(self):
        self.configservice = boto.connect_configservice()

    def test_describe_configuration_recorders(self):
        response = self.configservice.describe_configuration_recorders()
        self.assertIn('ConfigurationRecorders', response)

    def test_handle_no_such_configuration_recorder(self):
        with self.assertRaises(NoSuchConfigurationRecorderException):
            self.configservice.describe_configuration_recorders(configuration_recorder_names=['non-existant-recorder'])

    def test_connect_to_non_us_east_1(self):
        self.configservice = boto.configservice.connect_to_region('us-west-2')
        response = self.configservice.describe_configuration_recorders()
        self.assertIn('ConfigurationRecorders', response)