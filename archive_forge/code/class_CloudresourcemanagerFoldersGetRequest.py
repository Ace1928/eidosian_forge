from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersGetRequest(_messages.Message):
    """A CloudresourcemanagerFoldersGetRequest object.

  Fields:
    foldersId: Part of `name`. The resource name of the Folder to retrieve.
      Must be of the form `folders/{folder_id}`.
  """
    foldersId = _messages.StringField(1, required=True)