from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateFolderMetadata(_messages.Message):
    """Metadata pertaining to the Folder creation process.

  Fields:
    displayName: The display name of the folder.
    parent: The resource name of the folder or organization we are creating
      the folder under.
  """
    displayName = _messages.StringField(1)
    parent = _messages.StringField(2)