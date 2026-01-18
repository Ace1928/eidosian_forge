from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.resource_policies import flags
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def _ParseWeeklyDayAndTime(start_time, weekday):
    """Converts the dt and day to _API_TIMEZONE and returns API formatted values.

  Args:
    start_time: The datetime object which represents a start time.
    weekday: The times.Weekday value which corresponds to the weekday.

  Returns:
    The weekday and start_time pair formatted as strings for use by the API
    clients.
  """
    weekday = times.GetWeekdayInTimezone(start_time, weekday, _API_TIMEZONE)
    formatted_time = _FormatStartTime(start_time)
    return (weekday.name, formatted_time)