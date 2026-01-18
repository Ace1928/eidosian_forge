import os
from tests.unit import unittest
class TestCloudsearch2Connection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.cloudsearch2 import connect_to_region
        from boto.cloudsearch2.layer1 import CloudSearchConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, CloudSearchConnection)
        self.assertEqual(connection.host, 'cloudsearch.us-east-1.amazonaws.com')