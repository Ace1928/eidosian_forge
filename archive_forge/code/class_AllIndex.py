from boto.dynamodb2.types import STRING
class AllIndex(BaseIndexField):
    """
    An index signifying all fields should be in the index.

    Example::

        >>> AllIndex('MostRecentlyJoined', parts=[
        ...     HashKey('username'),
        ...     RangeKey('date_joined')
        ... ])

    """
    projection_type = 'ALL'