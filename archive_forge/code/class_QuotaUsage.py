from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaUsage(_messages.Message):
    """Specifies the used quota amount for a quota limit at a particular time.

  Fields:
    endTime: The time the quota duration ended.
    queryTime: The time the quota usage data was queried.
    startTime: The time the quota duration started.
    usage: The used quota value at the "query_time".
  """
    endTime = _messages.StringField(1)
    queryTime = _messages.StringField(2)
    startTime = _messages.StringField(3)
    usage = _messages.IntegerField(4)