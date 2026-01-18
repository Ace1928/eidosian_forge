from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def _MakeTagField(self, field_type, field_value):
    """Create a tag field."""
    value = self.messages.GoogleCloudDatacatalogV1beta1TagField()
    if field_type == 'double':
        value.doubleValue = field_value
    elif field_type == 'string':
        value.stringValue = field_value
    elif field_type == 'bool':
        value.boolValue = field_value
    elif field_type == 'timestamp':
        try:
            value.timestampValue = times.FormatDateTime(times.ParseDateTime(field_value))
        except times.Error as e:
            raise InvalidTagError('Invalid timestamp value: {} [{}]'.format(e, field_value))
    elif field_type == 'enum':
        value.enumValue = self.messages.GoogleCloudDatacatalogV1beta1TagFieldEnumValue(displayName=field_value)
    else:
        raise ValueError('Unknown field type: [{}]'.format(field_type))
    return value