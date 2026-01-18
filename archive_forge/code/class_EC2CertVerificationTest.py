import unittest
from tests.integration import ServiceCertVerificationTest
import boto.ec2
class EC2CertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    ec2 = True
    regions = boto.ec2.regions()

    def sample_service_call(self, conn):
        conn.get_all_reservations()