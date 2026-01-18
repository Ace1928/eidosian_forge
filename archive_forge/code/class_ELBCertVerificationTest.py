from tests.integration import ServiceCertVerificationTest
from tests.compat import unittest
import boto.ec2.elb
class ELBCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    elb = True
    regions = boto.ec2.elb.regions()

    def sample_service_call(self, conn):
        conn.get_all_load_balancers()