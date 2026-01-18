from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataFormatOptions(_messages.Message):
    """Options for data format adjustments.

  Fields:
    useInt64Timestamp: Optional. Output timestamp as usec int64. Default is
      false.
  """
    useInt64Timestamp = _messages.BooleanField(1)