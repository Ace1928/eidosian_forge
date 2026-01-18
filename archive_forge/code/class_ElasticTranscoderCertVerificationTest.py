from tests.integration import ServiceCertVerificationTest
import boto.elastictranscoder
from tests.compat import unittest
class ElasticTranscoderCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    elastictranscoder = True
    regions = boto.elastictranscoder.regions()

    def sample_service_call(self, conn):
        conn.list_pipelines()