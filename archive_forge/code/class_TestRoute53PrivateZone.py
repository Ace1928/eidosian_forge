import time
from tests.compat import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
from boto.exception import TooManyRecordsException
from boto.vpc import VPCConnection
@attr(route53=True)
class TestRoute53PrivateZone(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        time_str = str(int(time.time()))
        self.route53 = Route53Connection()
        self.base_domain = 'boto-private-zone-test-%s.com' % time_str
        self.vpc = VPCConnection()
        self.test_vpc = self.vpc.create_vpc(cidr_block='10.11.0.0/16')
        self.test_vpc.add_tag('Name', self.base_domain)
        self.zone = self.route53.get_zone(self.base_domain)
        if self.zone is not None:
            self.zone.delete()

    def test_create_private_zone(self):
        self.zone = self.route53.create_hosted_zone(self.base_domain, private_zone=True, vpc_id=self.test_vpc.id, vpc_region='us-east-1')

    @classmethod
    def tearDownClass(self):
        if self.zone is not None:
            self.zone.delete()
        self.test_vpc.delete()