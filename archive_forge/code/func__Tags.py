from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkemulticloud import util
from googlecloudsdk.command_lib.container.gkemulticloud import flags
def _Tags(self, args, parent_type):
    tags = flags.GetTags(args)
    if not tags:
        return None
    tag_type = parent_type.TagsValue.AdditionalProperty
    return parent_type.TagsValue(additionalProperties=[tag_type(key=k, value=v) for k, v in tags.items()])