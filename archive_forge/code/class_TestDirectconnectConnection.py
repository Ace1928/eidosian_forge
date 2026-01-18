import os
from tests.unit import unittest
class TestDirectconnectConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.directconnect import connect_to_region
        from boto.directconnect.layer1 import DirectConnectConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, DirectConnectConnection)
        self.assertEqual(connection.host, 'directconnect.us-east-1.amazonaws.com')