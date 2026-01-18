from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Customer(_messages.Message):
    """JSON template for Customer Resource object in Directory API.

  Fields:
    alternateEmail: The customer's secondary contact email address. This email
      address cannot be on the same domain as the customerDomain
    customerCreationTime: The customer's creation time (Readonly)
    customerDomain: The customer's primary domain name string. Do not include
      the www prefix when creating a new customer.
    etag: ETag of the resource.
    id: The unique ID for the customer's G Suite account. (Readonly)
    kind: Identifies the resource as a customer. Value:
      admin#directory#customer
    language: The customer's ISO 639-2 language code. The default value is en-
      US
    phoneNumber: The customer's contact phone number in E.164 format.
    postalAddress: The customer's postal address information.
  """
    alternateEmail = _messages.StringField(1)
    customerCreationTime = _message_types.DateTimeField(2)
    customerDomain = _messages.StringField(3)
    etag = _messages.StringField(4)
    id = _messages.StringField(5)
    kind = _messages.StringField(6, default=u'admin#directory#customer')
    language = _messages.StringField(7)
    phoneNumber = _messages.StringField(8)
    postalAddress = _messages.MessageField('CustomerPostalAddress', 9)