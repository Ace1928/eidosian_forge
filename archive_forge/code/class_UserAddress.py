from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserAddress(_messages.Message):
    """JSON template for address.

  Fields:
    country: Country.
    countryCode: Country code.
    customType: Custom type.
    extendedAddress: Extended Address.
    formatted: Formatted address.
    locality: Locality.
    poBox: Other parts of address.
    postalCode: Postal code.
    primary: If this is user's primary address. Only one entry could be marked
      as primary.
    region: Region.
    sourceIsStructured: User supplied address was structured. Structured
      addresses are NOT supported at this time. You might be able to write
      structured addresses, but any values will eventually be clobbered.
    streetAddress: Street.
    type: Each entry can have a type which indicates standard values of that
      entry. For example address could be of home, work etc. In addition to
      the standard type, an entry can have a custom type and can take any
      value. Such type should have the CUSTOM value as type and also have a
      customType value.
  """
    country = _messages.StringField(1)
    countryCode = _messages.StringField(2)
    customType = _messages.StringField(3)
    extendedAddress = _messages.StringField(4)
    formatted = _messages.StringField(5)
    locality = _messages.StringField(6)
    poBox = _messages.StringField(7)
    postalCode = _messages.StringField(8)
    primary = _messages.BooleanField(9)
    region = _messages.StringField(10)
    sourceIsStructured = _messages.BooleanField(11)
    streetAddress = _messages.StringField(12)
    type = _messages.StringField(13)