from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMembersInsertRequest(_messages.Message):
    """A DirectoryMembersInsertRequest object.

  Fields:
    groupKey: Email or immutable ID of the group
    member: A Member resource to be passed as the request body.
  """
    groupKey = _messages.StringField(1, required=True)
    member = _messages.MessageField('Member', 2)