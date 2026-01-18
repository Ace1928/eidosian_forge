from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsTopicsPublishRequest(_messages.Message):
    """A PubsubProjectsTopicsPublishRequest object.

  Fields:
    publishRequest: A PublishRequest resource to be passed as the request
      body.
    topic: The messages in the request will be published on this topic. Format
      is `projects/{project}/topics/{topic}`.
  """
    publishRequest = _messages.MessageField('PublishRequest', 1)
    topic = _messages.StringField(2, required=True)