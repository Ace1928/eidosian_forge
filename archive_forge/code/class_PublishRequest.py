from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PublishRequest(_messages.Message):
    """Request for the Publish method.

  Fields:
    messages: The messages to publish.
  """
    messages = _messages.MessageField('PubsubMessage', 1, repeated=True)