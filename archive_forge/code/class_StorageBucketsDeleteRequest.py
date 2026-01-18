from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageBucketsDeleteRequest(_messages.Message):
    """A StorageBucketsDeleteRequest object.

  Fields:
    bucket: Name of a bucket.
    ifMetagenerationMatch: If set, only deletes the bucket if its
      metageneration matches this value.
    ifMetagenerationNotMatch: If set, only deletes the bucket if its
      metageneration does not match this value.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    ifMetagenerationMatch = _messages.IntegerField(2)
    ifMetagenerationNotMatch = _messages.IntegerField(3)
    userProject = _messages.StringField(4)