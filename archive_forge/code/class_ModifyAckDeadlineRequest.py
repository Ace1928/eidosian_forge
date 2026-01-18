from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ModifyAckDeadlineRequest(_messages.Message):
    """Request for the ModifyAckDeadline method.

  Fields:
    ackDeadlineSeconds: The new ack deadline with respect to the time this
      request was sent to the Pub/Sub system. For example, if the value is 10,
      the new ack deadline will expire 10 seconds after the
      `ModifyAckDeadline` call was made. Specifying zero may immediately make
      the message available for another pull request. The minimum deadline you
      can specify is 0 seconds. The maximum deadline you can specify is 600
      seconds (10 minutes).
    ackIds: List of acknowledgment IDs.
  """
    ackDeadlineSeconds = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    ackIds = _messages.StringField(2, repeated=True)