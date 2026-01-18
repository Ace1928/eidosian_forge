from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
def GetPeeringZoneInfo(self, request, global_params=None):
    """Fetches the representation of an existing PeeringZone.

      Args:
        request: (DnsActivePeeringZonesGetPeeringZoneInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZone) The response message.
      """
    config = self.GetMethodConfig('GetPeeringZoneInfo')
    return self._RunMethod(config, request, global_params=global_params)