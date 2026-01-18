from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageProjectsHmacKeysGetRequest(_messages.Message):
    """A StorageProjectsHmacKeysGetRequest object.

  Fields:
    projectId: Project ID
    accessId: Name of the HMAC key for which the metadata is being requested.
  """
    projectId = _messages.StringField(2, required=True)
    accessId = _messages.StringField(1, required=True)