from boto.dynamodb2.types import STRING
class RangeKey(BaseSchemaField):
    """
    An field representing a range key.

    Example::

        >>> from boto.dynamodb2.types import NUMBER
        >>> HashKey('username')
        >>> HashKey('date_joined', data_type=NUMBER)

    """
    attr_type = 'RANGE'