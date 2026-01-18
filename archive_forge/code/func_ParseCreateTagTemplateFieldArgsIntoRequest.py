from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
def ParseCreateTagTemplateFieldArgsIntoRequest(self, args, request):
    """Parses tag-templates fields create args into the request."""
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1TagTemplateField.type', self._ParseFieldType(args.type))
    return request