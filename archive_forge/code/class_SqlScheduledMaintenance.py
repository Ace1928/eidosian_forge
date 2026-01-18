from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlScheduledMaintenance(_messages.Message):
    """Any scheduled maintenance for this instance.

  Fields:
    canDefer: A boolean attribute.
    canReschedule: If the scheduled maintenance can be rescheduled.
    scheduleDeadlineTime: Maintenance cannot be rescheduled to start beyond
      this deadline.
    startTime: The start time of any upcoming scheduled maintenance for this
      instance.
  """
    canDefer = _messages.BooleanField(1)
    canReschedule = _messages.BooleanField(2)
    scheduleDeadlineTime = _messages.StringField(3)
    startTime = _messages.StringField(4)