from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryAspsListRequest(_messages.Message):
    """A DirectoryAspsListRequest object.

  Fields:
    userKey: Identifies the user in the API request. The value can be the
      user's primary email address, alias email address, or unique user ID.
  """
    userKey = _messages.StringField(1, required=True)