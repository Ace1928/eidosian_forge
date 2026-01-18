import unittest
from tests.integration import ServiceCertVerificationTest
import boto.dynamodb
class DynamoDBCertVerificationTest(unittest.TestCase, ServiceCertVerificationTest):
    dynamodb = True
    regions = boto.dynamodb.regions()

    def sample_service_call(self, conn):
        conn.layer1.list_tables()