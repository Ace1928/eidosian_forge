import os
from tests.unit import unittest
class TestStsConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.sts import connect_to_region
        from boto.sts.connection import STSConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, STSConnection)
        self.assertEqual(connection.host, 'sts.amazonaws.com')