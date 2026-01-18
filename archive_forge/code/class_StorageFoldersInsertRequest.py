from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageFoldersInsertRequest(_messages.Message):
    """A StorageFoldersInsertRequest object.

  Fields:
    bucket: Name of the bucket in which the folder resides.
    folder: A Folder resource to be passed as the request body.
    recursive: If true, any parent folder which doesn't exist will be created
      automatically.
  """
    bucket = _messages.StringField(1, required=True)
    folder = _messages.MessageField('Folder', 2)
    recursive = _messages.BooleanField(3)