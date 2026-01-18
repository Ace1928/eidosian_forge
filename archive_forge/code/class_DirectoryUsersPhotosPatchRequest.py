from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryUsersPhotosPatchRequest(_messages.Message):
    """A DirectoryUsersPhotosPatchRequest object.

  Fields:
    userKey: Email or immutable ID of the user
    userPhoto: A UserPhoto resource to be passed as the request body.
  """
    userKey = _messages.StringField(1, required=True)
    userPhoto = _messages.MessageField('UserPhoto', 2)