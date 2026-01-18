from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterMaintenanceSchedule(_messages.Message):
    """Upcoming maitenance schedule.

  Fields:
    endTime: Output only. The end time of any upcoming scheduled maintenance
      for this instance.
    scheduleDeadlineTime: Output only. The deadline that the maintenance
      schedule start time can not go beyond, including reschedule.
    startTime: Output only. The start time of any upcoming scheduled
      maintenance for this instance.
  """
    endTime = _messages.StringField(1)
    scheduleDeadlineTime = _messages.StringField(2)
    startTime = _messages.StringField(3)