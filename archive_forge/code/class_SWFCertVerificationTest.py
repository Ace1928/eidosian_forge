import unittest
from tests.integration import ServiceCertVerificationTest
import boto.swf
class SWFCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    swf = True
    regions = boto.swf.regions()

    def sample_service_call(self, conn):
        conn.list_domains('REGISTERED')