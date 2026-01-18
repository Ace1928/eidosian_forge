import os
from tests.unit import unittest
class TestSqsConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.sqs import connect_to_region
        from boto.sqs.connection import SQSConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, SQSConnection)
        self.assertEqual(connection.host, 'queue.amazonaws.com')