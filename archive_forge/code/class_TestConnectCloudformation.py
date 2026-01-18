import os
from tests.unit import unittest
class TestConnectCloudformation(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.cloudformation import connect_to_region
        from boto.cloudformation.connection import CloudFormationConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CloudFormationConnection)
        self.assertEqual(connection.host, 'cloudformation.us-east-1.amazonaws.com')