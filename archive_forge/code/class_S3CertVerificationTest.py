import unittest
from tests.integration import ServiceCertVerificationTest
import boto.s3
class S3CertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    s3 = True
    regions = boto.s3.regions()

    def sample_service_call(self, conn):
        conn.get_all_buckets()