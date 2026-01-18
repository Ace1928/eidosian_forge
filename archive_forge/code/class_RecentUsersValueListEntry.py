from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecentUsersValueListEntry(_messages.Message):
    """A RecentUsersValueListEntry object.

    Fields:
      email: Email address of the user. Present only if the user type is
        managed
      type: The type of the user
    """
    email = _messages.StringField(1)
    type = _messages.StringField(2)