from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsListChannelsRequest(_messages.Message):
    """A StorageBucketsListChannelsRequest object.

  Fields:
    bucket: Name of a bucket.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    userProject = _messages.StringField(2)