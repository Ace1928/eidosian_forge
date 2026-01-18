from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryGroupsDeleteRequest(_messages.Message):
    """A DirectoryGroupsDeleteRequest object.

  Fields:
    groupKey: Email or immutable ID of the group
  """
    groupKey = _messages.StringField(1, required=True)