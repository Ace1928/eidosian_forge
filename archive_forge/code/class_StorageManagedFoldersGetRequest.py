from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageManagedFoldersGetRequest(_messages.Message):
    """A StorageManagedFoldersGetRequest object.

  Fields:
    bucket: Name of the bucket containing the managed folder.
    ifMetagenerationMatch: Makes the return of the managed folder metadata
      conditional on whether the managed folder's current metageneration
      matches the given value.
    ifMetagenerationNotMatch: Makes the return of the managed folder metadata
      conditional on whether the managed folder's current metageneration does
      not match the given value.
    managedFolder: The managed folder name/path.
  """
    bucket = _messages.StringField(1, required=True)
    ifMetagenerationMatch = _messages.IntegerField(2)
    ifMetagenerationNotMatch = _messages.IntegerField(3)
    managedFolder = _messages.StringField(4, required=True)