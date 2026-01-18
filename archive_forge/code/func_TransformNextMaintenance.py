from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformNextMaintenance(r, undefined=''):
    """Returns the timestamps of the next scheduled maintenance.

  All timestamps are assumed to be ISO strings in the same timezone.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    The timestamps of the next scheduled maintenance or undefined.
  """
    if not r:
        return undefined
    next_event = min(r, key=lambda x: x.get('beginTime', None))
    if next_event is None:
        return undefined
    begin_time = next_event.get('beginTime', None)
    if begin_time is None:
        return undefined
    end_time = next_event.get('endTime', None)
    if end_time is None:
        return undefined
    return '{0}--{1}'.format(begin_time, end_time)