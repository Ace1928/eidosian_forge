import unittest
from tests.integration import ServiceCertVerificationTest
import boto.ec2.autoscale
class AutoscaleCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    autoscale = True
    regions = boto.ec2.autoscale.regions()

    def sample_service_call(self, conn):
        conn.get_all_groups()