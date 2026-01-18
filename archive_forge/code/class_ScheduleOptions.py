from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScheduleOptions(_messages.Message):
    """Options customizing the data transfer schedule.

  Fields:
    disableAutoScheduling: If true, automatic scheduling of data transfer runs
      for this configuration will be disabled. The runs can be started on ad-
      hoc basis using StartManualTransferRuns API. When automatic scheduling
      is disabled, the TransferConfig.schedule field will be ignored.
    endTime: Defines time to stop scheduling transfer runs. A transfer run
      cannot be scheduled at or after the end time. The end time can be
      changed at any moment. The time when a data transfer can be triggered
      manually is not limited by this option.
    startTime: Specifies time to start scheduling transfer runs. The first run
      will be scheduled at or after the start time according to a recurrence
      pattern defined in the schedule string. The start time can be changed at
      any moment. The time when a data transfer can be triggered manually is
      not limited by this option.
  """
    disableAutoScheduling = _messages.BooleanField(1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)