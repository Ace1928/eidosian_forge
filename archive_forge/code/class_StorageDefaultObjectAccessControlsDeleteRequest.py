from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDefaultObjectAccessControlsDeleteRequest(_messages.Message):
    """A StorageDefaultObjectAccessControlsDeleteRequest object.

  Fields:
    bucket: Name of a bucket.
    entity: The entity holding the permission. Can be user-userId, user-
      emailAddress, group-groupId, group-emailAddress, allUsers, or
      allAuthenticatedUsers.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    entity = _messages.StringField(2, required=True)
    userProject = _messages.StringField(3)