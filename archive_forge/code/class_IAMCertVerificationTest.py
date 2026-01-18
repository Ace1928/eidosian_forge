import unittest
from tests.integration import ServiceCertVerificationTest
import boto.iam
class IAMCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    iam = True
    regions = boto.iam.regions()

    def sample_service_call(self, conn):
        conn.get_all_users()