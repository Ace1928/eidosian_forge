from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.vpc import VPCConnection, VPC
from boto.ec2.securitygroup import SecurityGroup
class TestVpcClassicLink(AWSMockServiceTestCase):
    connection_class = VPCConnection

    def setUp(self):
        super(TestVpcClassicLink, self).setUp()
        self.vpc = VPC(self.service_connection)
        self.vpc_id = 'myid'
        self.vpc.id = self.vpc_id