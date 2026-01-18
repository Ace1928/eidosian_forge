from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageManagedFoldersDeleteRequest(_messages.Message):
    """A StorageManagedFoldersDeleteRequest object.

  Fields:
    allowNonEmpty: Allows the deletion of a managed folder even if it is not
      empty. A managed folder is empty if there are no objects or managed
      folders that it applies to. Callers must have
      storage.managedFolders.setIamPolicy permission.
    bucket: Name of the bucket containing the managed folder.
    ifMetagenerationMatch: If set, only deletes the managed folder if its
      metageneration matches this value.
    ifMetagenerationNotMatch: If set, only deletes the managed folder if its
      metageneration does not match this value.
    managedFolder: The managed folder name/path.
  """
    allowNonEmpty = _messages.BooleanField(1)
    bucket = _messages.StringField(2, required=True)
    ifMetagenerationMatch = _messages.IntegerField(3)
    ifMetagenerationNotMatch = _messages.IntegerField(4)
    managedFolder = _messages.StringField(5, required=True)