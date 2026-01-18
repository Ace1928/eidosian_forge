from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsActivePeeringZonesDeactivateRequest(_messages.Message):
    """A DnsActivePeeringZonesDeactivateRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    peeringZoneId: The unique peering zone id of the consumer peering zone to
      be deactivated.
    project: The project ID for the producer project targeted by the consumer
      peering zone to be deactivated.
  """
    clientOperationId = _messages.StringField(1)
    peeringZoneId = _messages.IntegerField(2, required=True)
    project = _messages.StringField(3, required=True)