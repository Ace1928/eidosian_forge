from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AcknowledgeRequest(_messages.Message):
    """Request for the Acknowledge method.

  Fields:
    ackIds: The acknowledgment ID for the messages being acknowledged that was
      returned by the Pub/Sub system in the `Pull` response. Must not be
      empty.
  """
    ackIds = _messages.StringField(1, repeated=True)