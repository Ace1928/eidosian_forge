from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PickupInfo(_messages.Message):
    """Message containing pickup information for a return of an appliance.
  NextID: 8

  Fields:
    address: Required. The address to pick up the appliance from
    contactName: Required. The name of the customer site contact.
    phone: Required. The phone number of the customer site contact. Should be
      given in E.164 format consisting of the country calling code (1 to 3
      digits) and the subscriber number, with no additional spaces or
      formatting, e.g. "15552220123".
    pickupDate: Optional. Preferred pick up date requested by the customer.
    pickupInstructions: Optional. Pickup instructions provided by the
      customer.
    pickupTimeslot: Optional. Preferred pick up time slot requested by the
      customer.
    shippingLabelEmail: Optional. Emails to include when sending shipping
      labels.
  """
    address = _messages.MessageField('PostalAddress', 1)
    contactName = _messages.StringField(2)
    phone = _messages.StringField(3)
    pickupDate = _messages.MessageField('Date', 4)
    pickupInstructions = _messages.StringField(5)
    pickupTimeslot = _messages.StringField(6)
    shippingLabelEmail = _messages.StringField(7, repeated=True)