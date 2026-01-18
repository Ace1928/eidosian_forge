from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersUndeleteRequest(_messages.Message):
    """A DirectoryUsersUndeleteRequest object.

  Fields:
    userKey: The immutable id of the user
    userUndelete: A UserUndelete resource to be passed as the request body.
  """
    userKey = _messages.StringField(1, required=True)
    userUndelete = _messages.MessageField('UserUndelete', 2)