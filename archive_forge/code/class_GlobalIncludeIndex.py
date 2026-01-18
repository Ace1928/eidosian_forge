from boto.dynamodb2.types import STRING
class GlobalIncludeIndex(GlobalBaseIndexField, IncludeIndex):
    """
    An index signifying only certain fields should be in the index.

    Example::

        >>> GlobalIncludeIndex('GenderIndex', parts=[
        ...     HashKey('username'),
        ...     RangeKey('date_joined')
        ... ],
        ... includes=['gender'],
        ... throughput={
        ...     'read': 2,
        ...     'write': 1,
        ... })

    """
    projection_type = 'INCLUDE'

    def __init__(self, *args, **kwargs):
        throughput = kwargs.pop('throughput', None)
        IncludeIndex.__init__(self, *args, **kwargs)
        if throughput:
            kwargs['throughput'] = throughput
        GlobalBaseIndexField.__init__(self, *args, **kwargs)

    def schema(self):
        schema_data = IncludeIndex.schema(self)
        schema_data.update(GlobalBaseIndexField.schema(self))
        return schema_data