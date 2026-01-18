from boto.dynamodb2.types import STRING
class IncludeIndex(BaseIndexField):
    """
    An index signifying only certain fields should be in the index.

    Example::

        >>> IncludeIndex('GenderIndex', parts=[
        ...     HashKey('username'),
        ...     RangeKey('date_joined')
        ... ], includes=['gender'])

    """
    projection_type = 'INCLUDE'

    def __init__(self, *args, **kwargs):
        self.includes_fields = kwargs.pop('includes', [])
        super(IncludeIndex, self).__init__(*args, **kwargs)

    def schema(self):
        schema_data = super(IncludeIndex, self).schema()
        schema_data['Projection']['NonKeyAttributes'] = self.includes_fields
        return schema_data