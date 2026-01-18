from boto.dynamodb2.types import STRING
class BaseIndexField(object):
    """
    An abstract class for defining schema indexes.

    Contains most of the core functionality for the index. Subclasses must
    define a ``projection_type`` to pass to DynamoDB.
    """

    def __init__(self, name, parts):
        self.name = name
        self.parts = parts

    def definition(self):
        """
        Returns the attribute definition structure DynamoDB expects.

        Example::

            >>> index.definition()
            {
                'AttributeName': 'username',
                'AttributeType': 'S',
            }

        """
        definition = []
        for part in self.parts:
            definition.append({'AttributeName': part.name, 'AttributeType': part.data_type})
        return definition

    def schema(self):
        """
        Returns the schema structure DynamoDB expects.

        Example::

            >>> index.schema()
            {
                'IndexName': 'LastNameIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'username',
                        'KeyType': 'HASH',
                    },
                ],
                'Projection': {
                    'ProjectionType': 'KEYS_ONLY',
                }
            }

        """
        key_schema = []
        for part in self.parts:
            key_schema.append(part.schema())
        return {'IndexName': self.name, 'KeySchema': key_schema, 'Projection': {'ProjectionType': self.projection_type}}