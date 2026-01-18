from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
def ParseUpdateTagTemplateFieldArgsIntoRequest(self, args, request):
    """Parses tag-templates fields update args into the request."""
    enum_values = []
    if args.IsSpecified('enum_values'):
        for value in args.enum_values:
            enum_values.append(self._MakeEnumValue(value))
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1TagTemplateField.type', self.messages.GoogleCloudDatacatalogV1FieldType(enumType=self.messages.GoogleCloudDatacatalogV1FieldTypeEnumType(allowedValues=enum_values)))
    return request