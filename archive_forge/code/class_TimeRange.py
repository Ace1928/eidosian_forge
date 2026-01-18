from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeRange(_messages.Message):
    """A specification for a time range, this will request transfer runs with
  run_time between start_time (inclusive) and end_time (exclusive).

  Fields:
    endTime: End time of the range of transfer runs. For example,
      `"2017-05-30T00:00:00+00:00"`. The end_time must not be in the future.
      Creates transfer runs where run_time is in the range between start_time
      (inclusive) and end_time (exclusive).
    startTime: Start time of the range of transfer runs. For example,
      `"2017-05-25T00:00:00+00:00"`. The start_time must be strictly less than
      the end_time. Creates transfer runs where run_time is in the range
      between start_time (inclusive) and end_time (exclusive).
  """
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)