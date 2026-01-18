from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDefaultObjectAccessControlsInsertRequest(_messages.Message):
    """A StorageDefaultObjectAccessControlsInsertRequest object.

  Fields:
    bucket: Name of a bucket.
    objectAccessControl: A ObjectAccessControl resource to be passed as the
      request body.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    objectAccessControl = _messages.MessageField('ObjectAccessControl', 2)
    userProject = _messages.StringField(3)