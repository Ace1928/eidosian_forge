from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDefaultObjectAccessControlsListRequest(_messages.Message):
    """A StorageDefaultObjectAccessControlsListRequest object.

  Fields:
    bucket: Name of a bucket.
    ifMetagenerationMatch: If present, only return default ACL listing if the
      bucket's current metageneration matches this value.
    ifMetagenerationNotMatch: If present, only return default ACL listing if
      the bucket's current metageneration does not match the given value.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    ifMetagenerationMatch = _messages.IntegerField(2)
    ifMetagenerationNotMatch = _messages.IntegerField(3)
    userProject = _messages.StringField(4)