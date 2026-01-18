from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserName(_messages.Message):
    """JSON template for name of a user in Directory API.

  Fields:
    familyName: Last Name
    fullName: Full Name
    givenName: First Name
  """
    familyName = _messages.StringField(1)
    fullName = _messages.StringField(2)
    givenName = _messages.StringField(3)