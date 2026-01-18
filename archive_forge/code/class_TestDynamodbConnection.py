import os
from tests.unit import unittest
class TestDynamodbConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.dynamodb import connect_to_region
        from boto.dynamodb.layer2 import Layer2
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, Layer2)
        self.assertEqual(connection.layer1.host, 'dynamodb.us-east-1.amazonaws.com')