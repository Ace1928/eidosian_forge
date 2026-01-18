from mock import Mock
from tests.unit import unittest
import boto.ec2containerservice
from boto.ec2containerservice.layer1 import EC2ContainerServiceConnection
from boto.compat import http_client
class TestConnectToRegion(unittest.TestCase):

    def setUp(self):
        self.https_connection = Mock(spec=http_client.HTTPSConnection)
        self.https_connection_factory = (Mock(return_value=self.https_connection), ())

    def test_aws_region(self):
        ecs = boto.ec2containerservice.connect_to_region('us-east-1', https_connection_factory=self.https_connection_factory, aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
        self.assertIsInstance(ecs, EC2ContainerServiceConnection)