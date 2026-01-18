import unittest
from tests.integration import ServiceCertVerificationTest
import boto.kinesis
class KinesisCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    kinesis = True
    regions = boto.kinesis.regions()

    def sample_service_call(self, conn):
        conn.list_streams()