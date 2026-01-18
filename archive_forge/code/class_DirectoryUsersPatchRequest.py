from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersPatchRequest(_messages.Message):
    """A DirectoryUsersPatchRequest object.

  Fields:
    user: A User resource to be passed as the request body.
    userKey: Email or immutable ID of the user. If ID, it should match with id
      of user object
  """
    user = _messages.MessageField('User', 1)
    userKey = _messages.StringField(2, required=True)