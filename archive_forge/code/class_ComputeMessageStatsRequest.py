from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeMessageStatsRequest(_messages.Message):
    """Compute statistics about a range of messages in a given topic and
  partition.

  Fields:
    endCursor: The exclusive end of the range. The range is empty if
      end_cursor <= start_cursor. Specifying a start_cursor before the first
      message and an end_cursor after the last message will retrieve all
      messages.
    partition: Required. The partition for which we should compute message
      stats.
    startCursor: The inclusive start of the range.
  """
    endCursor = _messages.MessageField('Cursor', 1)
    partition = _messages.IntegerField(2)
    startCursor = _messages.MessageField('Cursor', 3)