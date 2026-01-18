from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersMakeAdminRequest(_messages.Message):
    """A DirectoryUsersMakeAdminRequest object.

  Fields:
    userKey: Email or immutable ID of the user as admin
    userMakeAdmin: A UserMakeAdmin resource to be passed as the request body.
  """
    userKey = _messages.StringField(1, required=True)
    userMakeAdmin = _messages.MessageField('UserMakeAdmin', 2)