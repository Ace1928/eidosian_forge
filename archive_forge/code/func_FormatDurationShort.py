from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.run.printers import container_and_volume_printer_util as container_util
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core.resource import custom_printer_base as cp
def FormatDurationShort(duration_seconds: int) -> str:
    """Format duration from seconds into shorthand string.

  Duration will be represented of the form `#d#h#m$s` for days, hours, minutes
  and seconds. Any field that's 0 will be excluded. So 3660 seconds (1 hour and
  1 minute) will be represented as "1h1m" with no days or seconds listed.

  Args:
    duration_seconds: the total time in seconds to format

  Returns:
    a string representing the duration in more human-friendly units.
  """
    if duration_seconds == 0:
        return '0s'
    duration = datetime.timedelta(seconds=duration_seconds)
    remaining = duration.seconds
    hours = remaining // 3600
    remaining = remaining % 3600
    minutes = remaining // 60
    seconds = remaining % 60
    res = ''
    if duration.days:
        res += '{}d'.format(duration.days)
    if hours:
        res += '{}h'.format(hours)
    if minutes:
        res += '{}m'.format(minutes)
    if seconds:
        res += '{}s'.format(seconds)
    return res