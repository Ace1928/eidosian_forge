from boto.exception import JSONResponseError
from boto.opsworks import connect_to_region, regions, RegionInfo
from boto.opsworks.layer1 import OpsWorksConnection
from tests.compat import unittest
class TestOpsWorksConnection(unittest.TestCase):
    opsworks = True

    def setUp(self):
        self.api = OpsWorksConnection()

    def test_describe_stacks(self):
        response = self.api.describe_stacks()
        self.assertIn('Stacks', response)

    def test_validation_errors(self):
        with self.assertRaises(JSONResponseError):
            self.api.create_stack('testbotostack', 'us-east-1', 'badarn', 'badarn2')