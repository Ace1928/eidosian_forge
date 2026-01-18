from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScheduleTransferRunsRequest(_messages.Message):
    """A request to schedule transfer runs for a time range.

  Fields:
    endTime: Required. End time of the range of transfer runs. For example,
      `"2017-05-30T00:00:00+00:00"`.
    startTime: Required. Start time of the range of transfer runs. For
      example, `"2017-05-25T00:00:00+00:00"`.
  """
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)