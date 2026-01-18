from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserWebsite(_messages.Message):
    """JSON template for a website entry.

  Fields:
    customType: Custom Type.
    primary: If this is user's primary website or not.
    type: Each entry can have a type which indicates standard types of that
      entry. For example website could be of home, work, blog etc. In addition
      to the standard type, an entry can have a custom type and can give it
      any name. Such types should have the CUSTOM value as type and also have
      a customType value.
    value: Website.
  """
    customType = _messages.StringField(1)
    primary = _messages.BooleanField(2)
    type = _messages.StringField(3)
    value = _messages.StringField(4)