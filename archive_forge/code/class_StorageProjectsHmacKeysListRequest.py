from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageProjectsHmacKeysListRequest(_messages.Message):
    """A StorageProjectsHmacKeysListRequest object.

  Fields:
    projectId: Project ID
    serviceAccountEmail: If present only, keys for the given service account
      will be returned.
    showDeletedKeys: Whether or not to show keys in the DELETED state.
    maxResults: Maximum number of items to return in a single
      page of responses. The service will use this
      parameter or 1,000 items, whichever is smaller.
    pageToken: A previously-returned page token representing part of the
      larger set of results to view.

  """
    projectId = _messages.StringField(1, required=True)
    serviceAccountEmail = _messages.StringField(2)
    showDeletedKeys = _messages.BooleanField(3)
    maxResults = _messages.IntegerField(4, variant=_messages.Variant.UINT32, default=1000)
    pageToken = _messages.StringField(5)