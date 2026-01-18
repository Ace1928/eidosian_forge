from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import tag_utils
def GetResourceManagerTags(resource_manager_tags):
    """Returns a map of resource manager tags, translating namespaced tags if needed.

  Args:
    resource_manager_tags: Map of resource manager tag key value pairs with
      either namespaced name or name.

  Returns:
    Map of resource manager tags with format tagKeys/[0-9]+, tagValues/[0-9]+
  """
    ret_resource_manager_tags = {}
    for key, value in resource_manager_tags.items():
        if not key.startswith('tagKeys/'):
            key = tag_utils.GetNamespacedResource(key, tag_utils.TAG_KEYS).name
        if not value.startswith('tagValues/'):
            value = tag_utils.GetNamespacedResource(value, tag_utils.TAG_VALUES).name
        ret_resource_manager_tags[key] = value
    return ret_resource_manager_tags