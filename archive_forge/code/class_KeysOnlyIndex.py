from boto.dynamodb2.types import STRING
class KeysOnlyIndex(BaseIndexField):
    """
    An index signifying only key fields should be in the index.

    Example::

        >>> KeysOnlyIndex('MostRecentlyJoined', parts=[
        ...     HashKey('username'),
        ...     RangeKey('date_joined')
        ... ])

    """
    projection_type = 'KEYS_ONLY'