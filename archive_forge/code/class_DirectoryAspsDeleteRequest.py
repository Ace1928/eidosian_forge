from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryAspsDeleteRequest(_messages.Message):
    """A DirectoryAspsDeleteRequest object.

  Fields:
    codeId: The unique ID of the ASP to be deleted.
    userKey: Identifies the user in the API request. The value can be the
      user's primary email address, alias email address, or unique user ID.
  """
    codeId = _messages.IntegerField(1, required=True, variant=_messages.Variant.INT32)
    userKey = _messages.StringField(2, required=True)