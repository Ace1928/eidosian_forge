from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamConfig(_messages.Message):
    """Describes the optional configuration payload that the customer wants to
  set up with for the instance.

  Fields:
    fallbackUri: User-specified fallback uri that should be launched from the
      client when there is a streaming server stock-out.
  """
    fallbackUri = _messages.StringField(1)