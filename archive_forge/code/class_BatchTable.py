import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
class BatchTable(object):
    """
    Used by ``Table`` as the context manager for batch writes.

    You likely don't want to try to use this object directly.
    """

    def __init__(self, table):
        self.table = table
        self._to_put = []
        self._to_delete = []
        self._unprocessed = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._to_put or self._to_delete:
            self.flush()
        if self._unprocessed:
            self.resend_unprocessed()

    def put_item(self, data, overwrite=False):
        self._to_put.append(data)
        if self.should_flush():
            self.flush()

    def delete_item(self, **kwargs):
        self._to_delete.append(kwargs)
        if self.should_flush():
            self.flush()

    def should_flush(self):
        if len(self._to_put) + len(self._to_delete) == 25:
            return True
        return False

    def flush(self):
        batch_data = {self.table.table_name: []}
        for put in self._to_put:
            item = Item(self.table, data=put)
            batch_data[self.table.table_name].append({'PutRequest': {'Item': item.prepare_full()}})
        for delete in self._to_delete:
            batch_data[self.table.table_name].append({'DeleteRequest': {'Key': self.table._encode_keys(delete)}})
        resp = self.table.connection.batch_write_item(batch_data)
        self.handle_unprocessed(resp)
        self._to_put = []
        self._to_delete = []
        return True

    def handle_unprocessed(self, resp):
        if len(resp.get('UnprocessedItems', [])):
            table_name = self.table.table_name
            unprocessed = resp['UnprocessedItems'].get(table_name, [])
            msg = '%s items were unprocessed. Storing for later.'
            boto.log.info(msg % len(unprocessed))
            self._unprocessed.extend(unprocessed)

    def resend_unprocessed(self):
        boto.log.info('Re-sending %s unprocessed items.' % len(self._unprocessed))
        while len(self._unprocessed):
            to_resend = self._unprocessed[:25]
            self._unprocessed = self._unprocessed[25:]
            batch_data = {self.table.table_name: to_resend}
            boto.log.info('Sending %s items' % len(to_resend))
            resp = self.table.connection.batch_write_item(batch_data)
            self.handle_unprocessed(resp)
            boto.log.info('%s unprocessed items left' % len(self._unprocessed))