from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import tag_templates
from googlecloudsdk.api_lib.data_catalog import tag_templates_v1
def UpdateCreateTagTemplateRequestWithInputV1(unused_ref, args, request):
    """Hook for updating request with flags for tag-templates create."""
    del unused_ref
    client = tag_templates_v1.TagTemplatesClient()
    return client.ParseCreateTagTemplateArgsIntoRequest(args, request)