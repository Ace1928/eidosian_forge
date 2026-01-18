from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShuffleWriteMetrics(_messages.Message):
    """Shuffle data written by task.

  Fields:
    bytesWritten: A string attribute.
    recordsWritten: A string attribute.
    writeTimeNanos: A string attribute.
  """
    bytesWritten = _messages.IntegerField(1)
    recordsWritten = _messages.IntegerField(2)
    writeTimeNanos = _messages.IntegerField(3)