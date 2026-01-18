from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadOnly(_messages.Message):
    """Options for a transaction that can only be used to read documents.

  Fields:
    readTime: Reads documents at the given time. This must be a microsecond
      precision timestamp within the past one hour, or if Point-in-Time
      Recovery is enabled, can additionally be a whole minute timestamp within
      the past 7 days.
  """
    readTime = _messages.StringField(1)