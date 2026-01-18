import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def _batch_get(self, keys, consistent=False, attributes=None):
    """
        The internal method that performs the actual batch get. Used extensively
        by ``BatchGetResultSet`` to perform each (paginated) request.
        """
    items = {self.table_name: {'Keys': []}}
    if consistent:
        items[self.table_name]['ConsistentRead'] = True
    if attributes is not None:
        items[self.table_name]['AttributesToGet'] = attributes
    for key_data in keys:
        raw_key = {}
        for key, value in key_data.items():
            raw_key[key] = self._dynamizer.encode(value)
        items[self.table_name]['Keys'].append(raw_key)
    raw_results = self.connection.batch_get_item(request_items=items)
    results = []
    unprocessed_keys = []
    for raw_item in raw_results['Responses'].get(self.table_name, []):
        item = Item(self)
        item.load({'Item': raw_item})
        results.append(item)
    raw_unprocessed = raw_results.get('UnprocessedKeys', {}).get(self.table_name, {})
    for raw_key in raw_unprocessed.get('Keys', []):
        py_key = {}
        for key, value in raw_key.items():
            py_key[key] = self._dynamizer.decode(value)
        unprocessed_keys.append(py_key)
    return {'results': results, 'last_key': None, 'unprocessed_keys': unprocessed_keys}