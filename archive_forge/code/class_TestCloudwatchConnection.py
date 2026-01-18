import os
from tests.unit import unittest
class TestCloudwatchConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.ec2.cloudwatch import connect_to_region
        from boto.ec2.cloudwatch import CloudWatchConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CloudWatchConnection)
        self.assertEqual(connection.host, 'monitoring.us-east-1.amazonaws.com')