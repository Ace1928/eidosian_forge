import os
from tests.unit import unittest
class TestSdbConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.sdb import connect_to_region
        from boto.sdb.connection import SDBConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, SDBConnection)
        self.assertEqual(connection.host, 'sdb.amazonaws.com')