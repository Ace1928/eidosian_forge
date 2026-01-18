from boto.compat import six
class BatchList(list):
    """
    A subclass of a list object that contains a collection of
    :class:`boto.dynamodb.batch.Batch` objects.
    """

    def __init__(self, layer2):
        list.__init__(self)
        self.unprocessed = None
        self.layer2 = layer2

    def add_batch(self, table, keys, attributes_to_get=None, consistent_read=False):
        """
        Add a Batch to this BatchList.

        :type table: :class:`boto.dynamodb.table.Table`
        :param table: The Table object in which the items are contained.

        :type keys: list
        :param keys: A list of scalar or tuple values.  Each element in the
            list represents one Item to retrieve.  If the schema for the
            table has both a HashKey and a RangeKey, each element in the
            list should be a tuple consisting of (hash_key, range_key).  If
            the schema for the table contains only a HashKey, each element
            in the list should be a scalar value of the appropriate type
            for the table schema. NOTE: The maximum number of items that
            can be retrieved for a single operation is 100. Also, the
            number of items retrieved is constrained by a 1 MB size limit.

        :type attributes_to_get: list
        :param attributes_to_get: A list of attribute names.
            If supplied, only the specified attribute names will
            be returned.  Otherwise, all attributes will be returned.
        """
        self.append(Batch(table, keys, attributes_to_get, consistent_read))

    def resubmit(self):
        """
        Resubmit the batch to get the next result set. The request object is
        rebuild from scratch meaning that all batch added between ``submit``
        and ``resubmit`` will be lost.

        Note: This method is experimental and subject to changes in future releases
        """
        del self[:]
        if not self.unprocessed:
            return None
        for table_name, table_req in six.iteritems(self.unprocessed):
            table_keys = table_req['Keys']
            table = self.layer2.get_table(table_name)
            keys = []
            for key in table_keys:
                h = key['HashKeyElement']
                r = None
                if 'RangeKeyElement' in key:
                    r = key['RangeKeyElement']
                keys.append((h, r))
            attributes_to_get = None
            if 'AttributesToGet' in table_req:
                attributes_to_get = table_req['AttributesToGet']
            self.add_batch(table, keys, attributes_to_get=attributes_to_get)
        return self.submit()

    def submit(self):
        res = self.layer2.batch_get_item(self)
        if 'UnprocessedKeys' in res:
            self.unprocessed = res['UnprocessedKeys']
        return res

    def to_dict(self):
        """
        Convert a BatchList object into format required for Layer1.
        """
        d = {}
        for batch in self:
            b = batch.to_dict()
            if b['Keys']:
                d[batch.table.name] = b
        return d