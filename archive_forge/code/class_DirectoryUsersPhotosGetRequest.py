from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersPhotosGetRequest(_messages.Message):
    """A DirectoryUsersPhotosGetRequest object.

  Fields:
    userKey: Email or immutable ID of the user
  """
    userKey = _messages.StringField(1, required=True)