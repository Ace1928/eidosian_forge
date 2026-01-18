from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserIm(_messages.Message):
    """JSON template for instant messenger of a user.

  Fields:
    customProtocol: Custom protocol.
    customType: Custom type.
    im: Instant messenger id.
    primary: If this is user's primary im. Only one entry could be marked as
      primary.
    protocol: Protocol used in the instant messenger. It should be one of the
      values from ImProtocolTypes map. Similar to type, it can take a CUSTOM
      value and specify the custom name in customProtocol field.
    type: Each entry can have a type which indicates standard types of that
      entry. For example instant messengers could be of home, work etc. In
      addition to the standard type, an entry can have a custom type and can
      take any value. Such types should have the CUSTOM value as type and also
      have a customType value.
  """
    customProtocol = _messages.StringField(1)
    customType = _messages.StringField(2)
    im = _messages.StringField(3)
    primary = _messages.BooleanField(4)
    protocol = _messages.StringField(5)
    type = _messages.StringField(6)