from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShareSettingsFolderConfig(_messages.Message):
    """Config for each folder in the share settings.

  Fields:
    folderId: The folder ID, should be same as the key of this folder config
      in the parent map. Folder id should be a string of number, and without
      "folders/" prefix.
  """
    folderId = _messages.StringField(1)