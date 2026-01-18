import unittest
from tests.integration import ServiceCertVerificationTest
import boto.redshift
class RedshiftCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    redshift = True
    regions = boto.redshift.regions()

    def sample_service_call(self, conn):
        conn.describe_cluster_versions()