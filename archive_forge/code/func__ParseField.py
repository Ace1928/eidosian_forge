from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
def _ParseField(self, field):
    """Parses a field."""
    key = field['id']
    value = self.messages.GoogleCloudDatacatalogV1TagTemplateField(displayName=field.get('display-name', None), type=self._ParseFieldType(field['type']), isRequired=field.get('required', False))
    return self.messages.GoogleCloudDatacatalogV1TagTemplate.FieldsValue.AdditionalProperty(key=key, value=value)