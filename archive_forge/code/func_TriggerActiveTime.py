from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def TriggerActiveTime(event_type, update_time):
    """Computes the time by which the trigger will become active.

  Args:
    event_type: str, the trigger's event type.
    update_time: str, the time when the trigger was last modified.

  Returns:
    The active time as a string, or None if the trigger is already active.
  """
    if not types.IsAuditLogType(event_type):
        return None
    update_dt = times.ParseDateTime(update_time)
    delay = iso_duration.Duration(minutes=MAX_ACTIVE_DELAY_MINUTES)
    active_dt = times.GetDateTimePlusDuration(update_dt, delay)
    if times.Now() >= active_dt:
        return None
    return times.FormatDateTime(active_dt, fmt='%H:%M:%S', tzinfo=times.LOCAL)