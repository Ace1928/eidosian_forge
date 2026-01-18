from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ParseCreateTagArgsIntoRequest(self, args, request):
    """Parses tag-templates create args into the request."""
    tag_template_ref = args.CONCEPTS.tag_template.Parse()
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1beta1Tag.template', tag_template_ref.RelativeName())
    arg_utils.SetFieldInMessage(request, 'googleCloudDatacatalogV1beta1Tag.fields', self._ProcessTagFromFile(tag_template_ref, args.tag_file))
    return request