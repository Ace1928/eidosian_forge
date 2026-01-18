from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsTopicsDeleteRequest(_messages.Message):
    """A PubsubProjectsTopicsDeleteRequest object.

  Fields:
    topic: Name of the topic to delete. Format is
      `projects/{project}/topics/{topic}`.
  """
    topic = _messages.StringField(1, required=True)