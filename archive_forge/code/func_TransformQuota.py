from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformQuota(r, undefined=''):
    """Formats a quota as usage/limit.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    The quota in r as usage/limit.
  """
    usage = resource_transform.GetKeyValue(r, 'usage', None)
    if usage is None:
        return undefined
    limit = resource_transform.GetKeyValue(r, 'limit', None)
    if limit is None:
        return undefined
    try:
        if usage == int(usage) and limit == int(limit):
            return '{0}/{1}'.format(int(usage), int(limit))
        return '{0:.2f}/{1:.2f}'.format(usage, limit)
    except (TypeError, ValueError):
        pass
    return undefined