from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveFolderMetadata(_messages.Message):
    """Metadata pertaining to the folder move process.

  Fields:
    destinationParent: The resource name of the folder or organization to move
      the folder to.
    displayName: The display name of the folder.
    sourceParent: The resource name of the folder's parent.
  """
    destinationParent = _messages.StringField(1)
    displayName = _messages.StringField(2)
    sourceParent = _messages.StringField(3)