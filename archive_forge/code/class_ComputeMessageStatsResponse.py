from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeMessageStatsResponse(_messages.Message):
    """Response containing stats for messages in the requested topic and
  partition.

  Fields:
    messageBytes: The number of quota bytes accounted to these messages.
    messageCount: The count of messages.
    minimumEventTime: The minimum event timestamp across these messages. For
      the purposes of this computation, if a message does not have an event
      time, we use the publish time. The timestamp will be unset if there are
      no messages.
    minimumPublishTime: The minimum publish timestamp across these messages.
      Note that publish timestamps within a partition are not guaranteed to be
      non-decreasing. The timestamp will be unset if there are no messages.
  """
    messageBytes = _messages.IntegerField(1)
    messageCount = _messages.IntegerField(2)
    minimumEventTime = _messages.StringField(3)
    minimumPublishTime = _messages.StringField(4)