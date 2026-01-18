from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformLocation(r, undefined=''):
    """Return the region or zone name.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    The region or zone name.
  """
    for scope in ('zone', 'region'):
        location = resource_transform.GetKeyValue(r, scope, None)
        if location:
            return resource_transform.TransformBaseName(location, undefined)
    return undefined