from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersDeleteRequest(_messages.Message):
    """A CloudresourcemanagerFoldersDeleteRequest object.

  Fields:
    foldersId: Part of `name`. the resource name of the Folder to be deleted.
      Must be of the form `folders/{folder_id}`.
  """
    foldersId = _messages.StringField(1, required=True)