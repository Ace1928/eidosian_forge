from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformImageAlias(r, undefined=''):
    """Returns a comma-separated list of alias names for an image.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    A comma-separated list of alias names for the image in r.
  """
    name = resource_transform.GetKeyValue(r, 'name', None)
    if name is None:
        return undefined
    project = resource_transform.TransformScope(resource_transform.GetKeyValue(r, 'selfLink', ''), 'projects').split('/')[0]
    aliases = [alias for alias, value in constants.IMAGE_ALIASES.items() if name.startswith(value.name_prefix) and value.project == project]
    return ','.join(aliases)