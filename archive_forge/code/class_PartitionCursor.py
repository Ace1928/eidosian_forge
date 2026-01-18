from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartitionCursor(_messages.Message):
    """A pair of a Cursor and the partition it is for.

  Fields:
    cursor: The value of the cursor.
    partition: The partition this is for.
  """
    cursor = _messages.MessageField('Cursor', 1)
    partition = _messages.IntegerField(2)