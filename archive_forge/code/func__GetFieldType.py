from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def _GetFieldType(self, message_type):
    """Get a field type from a type."""
    primitive_type_enum = self.messages.GoogleCloudDatacatalogV1beta1FieldType.PrimitiveTypeValueValuesEnum
    primitive_types = {primitive_type_enum.DOUBLE: 'double', primitive_type_enum.STRING: 'string', primitive_type_enum.BOOL: 'bool', primitive_type_enum.TIMESTAMP: 'timestamp'}
    if message_type.primitiveType:
        if message_type.primitiveType in primitive_types:
            return primitive_types[message_type.primitiveType]
    elif message_type.enumType:
        return 'enum'
    raise ValueError('Unknown field type in message {}'.format(message_type))