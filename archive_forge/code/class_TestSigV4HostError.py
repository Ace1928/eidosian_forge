from tests.compat import mock
from tests.compat import unittest
from tests.unit import AWSMockServiceTestCase
from tests.unit import MockServiceWithConfigTestCase
from boto.connection import AWSAuthConnection
from boto.s3.connection import S3Connection, HostRequiredError
from boto.s3.connection import S3ResponseError, Bucket
class TestSigV4HostError(MockServiceWithConfigTestCase):
    connection_class = S3Connection

    def test_historical_behavior(self):
        self.assertEqual(self.service_connection._required_auth_capability(), ['hmac-v4-s3'])
        self.assertEqual(self.service_connection.host, 's3.amazonaws.com')

    def test_sigv4_opt_in(self):
        host_value = 's3.cn-north-1.amazonaws.com.cn'
        self.config = {'s3': {'use-sigv4': True}}
        conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more', host=host_value)
        self.assertEqual(conn._required_auth_capability(), ['hmac-v4-s3'])
        self.assertEqual(conn.host, host_value)
        self.config = {'s3': {'host': host_value, 'use-sigv4': True}}
        conn = self.connection_class(aws_access_key_id='less', aws_secret_access_key='more')
        self.assertEqual(conn._required_auth_capability(), ['hmac-v4-s3'])
        self.assertEqual(conn.host, host_value)