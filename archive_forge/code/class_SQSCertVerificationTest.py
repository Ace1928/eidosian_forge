import unittest
from tests.integration import ServiceCertVerificationTest
import boto.sqs
class SQSCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    sqs = True
    regions = boto.sqs.regions()

    def sample_service_call(self, conn):
        conn.get_all_queues()