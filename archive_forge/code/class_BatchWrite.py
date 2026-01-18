from boto.compat import six
class BatchWrite(object):
    """
    Used to construct a BatchWrite request.  Each BatchWrite object
    represents a collection of PutItem and DeleteItem requests for
    a single Table.

    :ivar table: The Table object from which the item is retrieved.

    :ivar puts: A list of :class:`boto.dynamodb.item.Item` objects
        that you want to write to DynamoDB.

    :ivar deletes: A list of scalar or tuple values.  Each element in the
        list represents one Item to delete.  If the schema for the
        table has both a HashKey and a RangeKey, each element in the
        list should be a tuple consisting of (hash_key, range_key).  If
        the schema for the table contains only a HashKey, each element
        in the list should be a scalar value of the appropriate type
        for the table schema.
    """

    def __init__(self, table, puts=None, deletes=None):
        self.table = table
        self.puts = puts or []
        self.deletes = deletes or []

    def to_dict(self):
        """
        Convert the Batch object into the format required for Layer1.
        """
        op_list = []
        for item in self.puts:
            d = {'Item': self.table.layer2.dynamize_item(item)}
            d = {'PutRequest': d}
            op_list.append(d)
        for key in self.deletes:
            if isinstance(key, tuple):
                hash_key, range_key = key
            else:
                hash_key = key
                range_key = None
            k = self.table.layer2.build_key_from_values(self.table.schema, hash_key, range_key)
            d = {'Key': k}
            op_list.append({'DeleteRequest': d})
        return (self.table.name, op_list)