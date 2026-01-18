from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScheduleTransferRunsResponse(_messages.Message):
    """A response to schedule transfer runs for a time range.

  Fields:
    runs: The transfer runs that were scheduled.
  """
    runs = _messages.MessageField('TransferRun', 1, repeated=True)