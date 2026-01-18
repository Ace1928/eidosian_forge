from boto.dynamodb2.types import STRING
class GlobalKeysOnlyIndex(GlobalBaseIndexField):
    """
    An index signifying only key fields should be in the index.

    Example::

        >>> GlobalKeysOnlyIndex('MostRecentlyJoined', parts=[
        ...     HashKey('username'),
        ...     RangeKey('date_joined')
        ... ],
        ... throughput={
        ...     'read': 2,
        ...     'write': 1,
        ... })

    """
    projection_type = 'KEYS_ONLY'