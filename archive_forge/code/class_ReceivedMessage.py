from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ReceivedMessage(_messages.Message):
    """A message and its corresponding acknowledgment ID.

  Fields:
    ackId: This ID can be used to acknowledge the received message.
    message: The message.
  """
    ackId = _messages.StringField(1)
    message = _messages.MessageField('PubsubMessage', 2)