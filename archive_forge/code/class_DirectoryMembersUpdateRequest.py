from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMembersUpdateRequest(_messages.Message):
    """A DirectoryMembersUpdateRequest object.

  Fields:
    groupKey: Email or immutable ID of the group. If ID, it should match with
      id of group object
    member: A Member resource to be passed as the request body.
    memberKey: Email or immutable ID of the user. If ID, it should match with
      id of member object
  """
    groupKey = _messages.StringField(1, required=True)
    member = _messages.MessageField('Member', 2)
    memberKey = _messages.StringField(3, required=True)