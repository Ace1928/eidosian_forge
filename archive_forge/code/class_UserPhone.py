from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserPhone(_messages.Message):
    """JSON template for a phone entry.

  Fields:
    customType: Custom Type.
    primary: If this is user's primary phone or not.
    type: Each entry can have a type which indicates standard types of that
      entry. For example phone could be of home_fax, work, mobile etc. In
      addition to the standard type, an entry can have a custom type and can
      give it any name. Such types should have the CUSTOM value as type and
      also have a customType value.
    value: Phone number.
  """
    customType = _messages.StringField(1)
    primary = _messages.BooleanField(2)
    type = _messages.StringField(3)
    value = _messages.StringField(4)