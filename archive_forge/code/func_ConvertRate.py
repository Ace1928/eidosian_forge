from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.tasks import constants
import six
def ConvertRate(value):
    """Converts the time based rate into its integer value in seconds.

  This function converts the input float values into its seconds equivalent.
  For example,
    '100/s' => 100.0
    '60/m' => 1.0

  Args:
    value: The string value we want to convert.

  Returns:
    A float value representing the rate on a per second basis
  """
    float_value, unit = (float(value[:-2]), value[-1])
    return round(float_value / constants.TIME_IN_SECONDS[unit], 9)