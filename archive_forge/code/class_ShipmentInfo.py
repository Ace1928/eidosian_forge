from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShipmentInfo(_messages.Message):
    """Message containing information about the appliance's shipment.

  Fields:
    deliverAfterTime: An optional timestamp to indicate the earliest that the
      appliance should be delivered. If not set it will be delivered as soon
      as possible.
    deliveryTrackingCarrier: Output only. Shipment carrier/partner associated
      with the outbound shipment (Google to customer).
    deliveryTrackingId: Output only. Tracking id associated with the outbound
      shipment (Google to customer).
    deliveryTrackingUri: Output only. A web URI allowing you to track the
      shipment from Google.
    returnLabelUri: Output only. The web URI to access the return label. This
      is only available once the finalization_code has been set on the
      Appliance resource.
    returnTrackingCarrier: Output only. Shipment carrier/partner associated
      with the inbound shipment (customer to Google).
    returnTrackingId: Output only. Tracking id associated with the inbound
      shipment (customer to Google).
    returnTrackingUri: Output only. A web URI allowing you to track the return
      shipment to Google.
  """
    deliverAfterTime = _messages.MessageField('DateTime', 1)
    deliveryTrackingCarrier = _messages.StringField(2)
    deliveryTrackingId = _messages.StringField(3)
    deliveryTrackingUri = _messages.StringField(4)
    returnLabelUri = _messages.StringField(5)
    returnTrackingCarrier = _messages.StringField(6)
    returnTrackingId = _messages.StringField(7)
    returnTrackingUri = _messages.StringField(8)