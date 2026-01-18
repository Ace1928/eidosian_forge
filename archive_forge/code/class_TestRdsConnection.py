import os
from tests.unit import unittest
class TestRdsConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.rds import connect_to_region
        from boto.rds import RDSConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, RDSConnection)
        self.assertEqual(connection.host, 'rds.amazonaws.com')