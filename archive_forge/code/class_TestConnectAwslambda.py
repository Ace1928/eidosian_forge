import os
from tests.unit import unittest
class TestConnectAwslambda(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.awslambda import connect_to_region
        from boto.awslambda.layer1 import AWSLambdaConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, AWSLambdaConnection)
        self.assertEqual(connection.host, 'lambda.us-east-1.amazonaws.com')