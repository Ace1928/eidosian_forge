from tests.unit import unittest
from boto.dynamodb.batch import Batch
from boto.dynamodb.table import Table
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.batch import BatchList
class TestBatchObjects(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.layer2 = Layer2('access_key', 'secret_key')
        self.table = Table(self.layer2, DESCRIBE_TABLE_1)
        self.table2 = Table(self.layer2, DESCRIBE_TABLE_2)

    def test_batch_to_dict(self):
        b = Batch(self.table, ['k1', 'k2'], attributes_to_get=['foo'], consistent_read=True)
        self.assertDictEqual(b.to_dict(), {'AttributesToGet': ['foo'], 'Keys': [{'HashKeyElement': {'S': 'k1'}}, {'HashKeyElement': {'S': 'k2'}}], 'ConsistentRead': True})

    def test_batch_consistent_read_defaults_to_false(self):
        b = Batch(self.table, ['k1'])
        self.assertDictEqual(b.to_dict(), {'Keys': [{'HashKeyElement': {'S': 'k1'}}], 'ConsistentRead': False})

    def test_batch_list_consistent_read(self):
        b = BatchList(self.layer2)
        b.add_batch(self.table, ['k1'], ['foo'], consistent_read=True)
        b.add_batch(self.table2, [('k2', 54)], ['bar'], consistent_read=False)
        self.assertDictEqual(b.to_dict(), {'testtable': {'AttributesToGet': ['foo'], 'Keys': [{'HashKeyElement': {'S': 'k1'}}], 'ConsistentRead': True}, 'testtable2': {'AttributesToGet': ['bar'], 'Keys': [{'HashKeyElement': {'S': 'k2'}, 'RangeKeyElement': {'N': '54'}}], 'ConsistentRead': False}})