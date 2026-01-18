from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeviceFilesValueListEntry(_messages.Message):
    """A DeviceFilesValueListEntry object.

    Fields:
      createTime: Date and time the file was created
      downloadUrl: File download URL
      name: File name
      type: File type
    """
    createTime = _message_types.DateTimeField(1)
    downloadUrl = _messages.StringField(2)
    name = _messages.StringField(3)
    type = _messages.StringField(4)