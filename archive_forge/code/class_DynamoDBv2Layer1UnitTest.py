from tests.unit import unittest
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.regioninfo import RegionInfo
class DynamoDBv2Layer1UnitTest(unittest.TestCase):
    dynamodb = True

    def test_init_region(self):
        dynamodb = DynamoDBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
        self.assertEqual(dynamodb.region.name, 'us-east-1')
        dynamodb = DynamoDBConnection(region=RegionInfo(name='us-west-2', endpoint='dynamodb.us-west-2.amazonaws.com'), aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')
        self.assertEqual(dynamodb.region.name, 'us-west-2')

    def test_init_host_override(self):
        dynamodb = DynamoDBConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', host='localhost', port=8000)
        self.assertEqual(dynamodb.host, 'localhost')
        self.assertEqual(dynamodb.port, 8000)