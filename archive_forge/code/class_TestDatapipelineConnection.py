import os
from tests.unit import unittest
class TestDatapipelineConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.datapipeline import connect_to_region
        from boto.datapipeline.layer1 import DataPipelineConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, DataPipelineConnection)
        self.assertEqual(connection.host, 'datapipeline.us-east-1.amazonaws.com')