import unittest
from tests.integration import ServiceCertVerificationTest
import boto.sns
class SNSCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    sns = True
    regions = boto.sns.regions()

    def sample_service_call(self, conn):
        conn.get_all_topics()