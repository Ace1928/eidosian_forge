from tests.compat import mock, unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import (STRING, NUMBER, BINARY,
from boto.exception import JSONResponseError
from boto.compat import six, long_type
class BatchGetResultSetTestCase(unittest.TestCase):

    def setUp(self):
        super(BatchGetResultSetTestCase, self).setUp()
        self.results = BatchGetResultSet(keys=['alice', 'bob', 'jane', 'johndoe'])
        self.results.to_call(fake_batch_results)

    def test_fetch_more(self):
        self.results.fetch_more()
        self.assertEqual(self.results._results, ['hello alice', 'hello bob', 'hello jane'])
        self.assertEqual(self.results._keys_left, ['johndoe'])
        self.results.fetch_more()
        self.assertEqual(self.results._results, ['hello johndoe'])
        self.results.fetch_more()
        self.assertEqual(self.results._results, [])
        self.assertFalse(self.results._results_left)

    def test_fetch_more_empty(self):
        self.results.to_call(lambda keys: {'results': [], 'last_key': None})
        self.results.fetch_more()
        self.assertEqual(self.results._results, [])
        self.assertRaises(StopIteration, self.results.next)

    def test_iteration(self):
        self.assertEqual(next(self.results), 'hello alice')
        self.assertEqual(next(self.results), 'hello bob')
        self.assertEqual(next(self.results), 'hello jane')
        self.assertEqual(next(self.results), 'hello johndoe')
        self.assertRaises(StopIteration, self.results.next)