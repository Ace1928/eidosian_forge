from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartManualTransferRunsRequest(_messages.Message):
    """A request to start manual transfer runs.

  Fields:
    requestedRunTime: A run_time timestamp for historical data files or
      reports that are scheduled to be transferred by the scheduled transfer
      run. requested_run_time must be a past time and cannot include future
      time values.
    requestedTimeRange: A time_range start and end timestamp for historical
      data files or reports that are scheduled to be transferred by the
      scheduled transfer run. requested_time_range must be a past time and
      cannot include future time values.
  """
    requestedRunTime = _messages.StringField(1)
    requestedTimeRange = _messages.MessageField('TimeRange', 2)