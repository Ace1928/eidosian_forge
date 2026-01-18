from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserGender(_messages.Message):
    """A UserGender object.

  Fields:
    addressMeAs: AddressMeAs. A human-readable string containing the proper
      way to refer to the profile owner by humans, for example "he/him/his" or
      "they/them/their".
    customGender: Custom gender.
    type: Gender.
  """
    addressMeAs = _messages.StringField(1)
    customGender = _messages.StringField(2)
    type = _messages.StringField(3)