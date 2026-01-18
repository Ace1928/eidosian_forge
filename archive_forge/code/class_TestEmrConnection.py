import os
from tests.unit import unittest
class TestEmrConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.emr import connect_to_region
        from boto.emr.connection import EmrConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, EmrConnection)
        self.assertEqual(connection.host, 'elasticmapreduce.us-east-1.amazonaws.com')