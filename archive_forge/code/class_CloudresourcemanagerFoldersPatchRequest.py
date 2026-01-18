from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerFoldersPatchRequest(_messages.Message):
    """A CloudresourcemanagerFoldersPatchRequest object.

  Fields:
    folder: A Folder resource to be passed as the request body.
    foldersId: Part of `folder.name`. Output only. The resource name of the
      Folder. Its format is `folders/{folder_id}`, for example:
      "folders/1234".
    updateMask: Fields to be updated. Only the `display_name` can be updated.
  """
    folder = _messages.MessageField('Folder', 1)
    foldersId = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)