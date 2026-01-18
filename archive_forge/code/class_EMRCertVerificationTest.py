import unittest
from tests.integration import ServiceCertVerificationTest
import boto.emr
class EMRCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    emr = True
    regions = boto.emr.regions()

    def sample_service_call(self, conn):
        conn.describe_jobflows()