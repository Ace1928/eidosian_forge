import os
from tests.unit import unittest
class TestRedshiftConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.redshift import connect_to_region
        from boto.redshift.layer1 import RedshiftConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, RedshiftConnection)
        self.assertEqual(connection.host, 'redshift.us-east-1.amazonaws.com')