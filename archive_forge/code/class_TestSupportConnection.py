import os
from tests.unit import unittest
class TestSupportConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.support import connect_to_region
        from boto.support.layer1 import SupportConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, SupportConnection)
        self.assertEqual(connection.host, 'support.us-east-1.amazonaws.com')