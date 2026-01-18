from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.tasks import constants
import six
def ConvertTaskAgeLimit(value):
    """Converts task age limit values to the format CT expects.

  Args:
    value: A string value representing the task age limit. For example, '2.5m',
      '1h', '8s', etc.

  Returns:
    The string representing the time to the nearest second with 's' appended
    at the end.
  """
    time_in_seconds = float(value[:-1]) * constants.TIME_IN_SECONDS[value[-1]]
    return '{}s'.format(int(time_in_seconds))