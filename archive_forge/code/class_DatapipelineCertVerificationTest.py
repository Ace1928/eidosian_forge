import unittest
from tests.integration import ServiceCertVerificationTest
import boto.datapipeline
class DatapipelineCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    datapipeline = True
    regions = boto.datapipeline.regions()

    def sample_service_call(self, conn):
        conn.list_pipelines()