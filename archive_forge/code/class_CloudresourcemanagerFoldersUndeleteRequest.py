from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersUndeleteRequest(_messages.Message):
    """A CloudresourcemanagerFoldersUndeleteRequest object.

  Fields:
    foldersId: Part of `name`. The resource name of the Folder to undelete.
      Must be of the form `folders/{folder_id}`.
    undeleteFolderRequest: A UndeleteFolderRequest resource to be passed as
      the request body.
  """
    foldersId = _messages.StringField(1, required=True)
    undeleteFolderRequest = _messages.MessageField('UndeleteFolderRequest', 2)