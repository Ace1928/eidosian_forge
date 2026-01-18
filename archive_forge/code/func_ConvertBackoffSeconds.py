from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.tasks import constants
import six
def ConvertBackoffSeconds(value):
    """Converts min/max backoff values to the format CT expects.

  Args:
    value: A float value representing time in seconds.

  Returns:
    The string representing the time with 's' appended at the end.
  """
    if value is None:
        return None
    return '{}s'.format(round(value, 8))