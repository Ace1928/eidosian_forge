from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesDnsZonesAddRequest(_messages.Message):
    """A ServicenetworkingServicesDnsZonesAddRequest object.

  Fields:
    addDnsZoneRequest: A AddDnsZoneRequest resource to be passed as the
      request body.
    parent: Required. The service that is managing peering connectivity for a
      service producer's organization. For Google services that support this
      functionality, this value is
      `services/servicenetworking.googleapis.com`.
  """
    addDnsZoneRequest = _messages.MessageField('AddDnsZoneRequest', 1)
    parent = _messages.StringField(2, required=True)