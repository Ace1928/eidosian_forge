from boto.compat import six
class BatchWriteList(list):
    """
    A subclass of a list object that contains a collection of
    :class:`boto.dynamodb.batch.BatchWrite` objects.
    """

    def __init__(self, layer2):
        list.__init__(self)
        self.layer2 = layer2

    def add_batch(self, table, puts=None, deletes=None):
        """
        Add a BatchWrite to this BatchWriteList.

        :type table: :class:`boto.dynamodb.table.Table`
        :param table: The Table object in which the items are contained.

        :type puts: list of :class:`boto.dynamodb.item.Item` objects
        :param puts: A list of items that you want to write to DynamoDB.

        :type deletes: A list
        :param deletes: A list of scalar or tuple values.  Each element
            in the list represents one Item to delete.  If the schema
            for the table has both a HashKey and a RangeKey, each
            element in the list should be a tuple consisting of
            (hash_key, range_key).  If the schema for the table
            contains only a HashKey, each element in the list should
            be a scalar value of the appropriate type for the table
            schema.
        """
        self.append(BatchWrite(table, puts, deletes))

    def submit(self):
        return self.layer2.batch_write_item(self)

    def to_dict(self):
        """
        Convert a BatchWriteList object into format required for Layer1.
        """
        d = {}
        for batch in self:
            table_name, batch_dict = batch.to_dict()
            d[table_name] = batch_dict
        return d