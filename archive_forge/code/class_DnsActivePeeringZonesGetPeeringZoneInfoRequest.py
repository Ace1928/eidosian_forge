from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsActivePeeringZonesGetPeeringZoneInfoRequest(_messages.Message):
    """A DnsActivePeeringZonesGetPeeringZoneInfoRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    peeringZoneId: ManagedZoneId addressed by this request
    project: Identifies the producer project targeted by the peering zone in
      this request.
  """
    clientOperationId = _messages.StringField(1)
    peeringZoneId = _messages.IntegerField(2, required=True)
    project = _messages.StringField(3, required=True)