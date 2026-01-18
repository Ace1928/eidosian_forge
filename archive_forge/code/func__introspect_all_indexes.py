import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def _introspect_all_indexes(self, raw_indexes, map_indexes_projection):
    """
        Given a raw index/global index structure back from a DynamoDB response,
        parse out & build the high-level Python objects that represent them.
        """
    indexes = []
    for field in raw_indexes:
        index_klass = map_indexes_projection.get('ALL')
        kwargs = {'parts': []}
        if field['Projection']['ProjectionType'] == 'ALL':
            index_klass = map_indexes_projection.get('ALL')
        elif field['Projection']['ProjectionType'] == 'KEYS_ONLY':
            index_klass = map_indexes_projection.get('KEYS_ONLY')
        elif field['Projection']['ProjectionType'] == 'INCLUDE':
            index_klass = map_indexes_projection.get('INCLUDE')
            kwargs['includes'] = field['Projection']['NonKeyAttributes']
        else:
            raise exceptions.UnknownIndexFieldError('%s was seen, but is unknown. Please report this at https://github.com/boto/boto/issues.' % field['Projection']['ProjectionType'])
        name = field['IndexName']
        kwargs['parts'] = self._introspect_schema(field['KeySchema'], None)
        indexes.append(index_klass(name, **kwargs))
    return indexes