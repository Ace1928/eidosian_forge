from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageProjectsHmacKeysCreateRequest(_messages.Message):
    """A StorageProjectsHmacKeysCreateRequest object.

  Fields:
    projectId: Project ID
    serviceAccountEmail: Email address of the service account for which to
      create a key.
  """
    projectId = _messages.StringField(1, required=True)
    serviceAccountEmail = _messages.StringField(2, required=True)