import unittest
from tests.integration import ServiceCertVerificationTest
import boto.glacier
class GlacierCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    glacier = True
    regions = boto.glacier.regions()

    def sample_service_call(self, conn):
        conn.list_vaults()