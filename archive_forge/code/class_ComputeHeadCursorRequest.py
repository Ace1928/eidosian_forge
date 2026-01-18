from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeHeadCursorRequest(_messages.Message):
    """Compute the current head cursor for a partition.

  Fields:
    partition: Required. The partition for which we should compute the head
      cursor.
  """
    partition = _messages.IntegerField(1)