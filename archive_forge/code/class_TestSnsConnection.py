import os
from tests.unit import unittest
class TestSnsConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.sns import connect_to_region
        from boto.sns.connection import SNSConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, SNSConnection)
        self.assertEqual(connection.host, 'sns.us-east-1.amazonaws.com')