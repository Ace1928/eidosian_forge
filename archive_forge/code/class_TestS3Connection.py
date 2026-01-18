import os
from tests.unit import unittest
class TestS3Connection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.s3 import connect_to_region
        from boto.s3.connection import S3Connection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, S3Connection)
        self.assertEqual(connection.host, 's3.amazonaws.com')

    def test_connect_to_custom_host(self):
        from boto.s3 import connect_to_region
        from boto.s3.connection import S3Connection
        host = 'mycustomhost.example.com'
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar', host=host)
        self.assertIsInstance(connection, S3Connection)
        self.assertEqual(connection.host, host)