from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PullResponse(_messages.Message):
    """Response for the `Pull` method.

  Fields:
    receivedMessages: Received Pub/Sub messages. The Pub/Sub system will
      return zero messages if there are no more available in the backlog. The
      Pub/Sub system may return fewer than the `maxMessages` requested even if
      there are more messages available in the backlog.
  """
    receivedMessages = _messages.MessageField('ReceivedMessage', 1, repeated=True)