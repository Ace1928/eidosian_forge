from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StructuredMessage(_messages.Message):
    """A rich message format, including a human readable string, a key for
  identifying the message, and structured data associated with the message for
  programmatic consumption.

  Fields:
    messageKey: Identifier for this message type. Used by external systems to
      internationalize or personalize message.
    messageText: Human-readable version of message.
    parameters: The structured data associated with this message.
  """
    messageKey = _messages.StringField(1)
    messageText = _messages.StringField(2)
    parameters = _messages.MessageField('Parameter', 3, repeated=True)