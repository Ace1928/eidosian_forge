import os
from tests.unit import unittest
class TestCodeDeployConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.codedeploy import connect_to_region
        from boto.codedeploy.layer1 import CodeDeployConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CodeDeployConnection)
        self.assertEqual(connection.host, 'codedeploy.us-east-1.amazonaws.com')