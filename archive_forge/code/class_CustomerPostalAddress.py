from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomerPostalAddress(_messages.Message):
    """JSON template for postal address of a customer.

  Fields:
    addressLine1: A customer's physical address. The address can be composed
      of one to three lines.
    addressLine2: Address line 2 of the address.
    addressLine3: Address line 3 of the address.
    contactName: The customer contact's name.
    countryCode: This is a required property. For countryCode information see
      the ISO 3166 country code elements.
    locality: Name of the locality. An example of a locality value is the city
      of San Francisco.
    organizationName: The company or company division name.
    postalCode: The postal code. A postalCode example is a postal zip code
      such as 10009. This is in accordance with - http://portablecontacts.net
      /draft-spec.html#address_element.
    region: Name of the region. An example of a region value is NY for the
      state of New York.
  """
    addressLine1 = _messages.StringField(1)
    addressLine2 = _messages.StringField(2)
    addressLine3 = _messages.StringField(3)
    contactName = _messages.StringField(4)
    countryCode = _messages.StringField(5)
    locality = _messages.StringField(6)
    organizationName = _messages.StringField(7)
    postalCode = _messages.StringField(8)
    region = _messages.StringField(9)