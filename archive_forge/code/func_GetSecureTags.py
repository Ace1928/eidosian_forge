from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.resource_manager import tag_utils
def GetSecureTags(secure_tags):
    """Returns a list of secure tags, translating namespaced tags if needed.

  Args:
    secure_tags: array of secure tag values with either namespaced name or name.

  Returns:
    List of secure tags with format tagValues/[0-9]+
  """
    ret_secure_tags = []
    for tag in secure_tags:
        if tag.startswith('tagValues/'):
            ret_secure_tags.append(tag)
        else:
            ret_secure_tags.append(tag_utils.GetNamespacedResource(tag, tag_utils.TAG_VALUES).name)
    return ret_secure_tags