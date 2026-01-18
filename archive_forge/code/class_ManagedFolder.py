from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedFolder(_messages.Message):
    """A managed folder.

  Fields:
    bucket: The name of the bucket containing this managed folder.
    createTime: The creation time of the managed folder in RFC 3339 format.
    id: The ID of the managed folder, including the bucket name and managed
      folder name.
    kind: The kind of item this is. For managed folders, this is always
      storage#managedFolder.
    metageneration: The version of the metadata for this managed folder. Used
      for preconditions and for detecting changes in metadata.
    name: The name of the managed folder. Required if not specified by URL
      parameter.
    selfLink: The link to this managed folder.
    updateTime: The last update time of the managed folder metadata in RFC
      3339 format.
  """
    bucket = _messages.StringField(1)
    createTime = _message_types.DateTimeField(2)
    id = _messages.StringField(3)
    kind = _messages.StringField(4, default='storage#managedFolder')
    metageneration = _messages.IntegerField(5)
    name = _messages.StringField(6)
    selfLink = _messages.StringField(7)
    updateTime = _message_types.DateTimeField(8)