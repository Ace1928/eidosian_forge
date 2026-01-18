from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PublishResponse(_messages.Message):
    """Response for the `Publish` method.

  Fields:
    messageIds: The server-assigned ID of each published message, in the same
      order as the messages in the request. IDs are guaranteed to be unique
      within the topic.
  """
    messageIds = _messages.StringField(1, repeated=True)