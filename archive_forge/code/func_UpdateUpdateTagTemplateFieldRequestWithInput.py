from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import tag_templates
from googlecloudsdk.api_lib.data_catalog import tag_templates_v1
def UpdateUpdateTagTemplateFieldRequestWithInput(unused_ref, args, request):
    """Hook for updating request with flags for tag-templates fields update."""
    del unused_ref
    update_mask = []
    if args.IsSpecified('display_name'):
        update_mask.append('display_name')
    if args.IsSpecified('enum_values'):
        update_mask.append('type.enum_type')
    if args.IsSpecified('required'):
        update_mask.append('is_required')
    request.updateMask = ','.join(update_mask)
    client = tag_templates.TagTemplatesClient()
    return client.ParseUpdateTagTemplateFieldArgsIntoRequest(args, request)