import os
from tests.unit import unittest
class TestEc2ContainerserviceConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.ec2containerservice import connect_to_region
        import boto.ec2containerservice.layer1 as layer1
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, layer1.EC2ContainerServiceConnection)
        self.assertEqual(connection.host, 'ecs.us-east-1.amazonaws.com')