from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
def Deactivate(self, request, global_params=None):
    """Deactivates a Peering Zone if it's not already deactivated. Returns an error if the managed zone cannot be found, is not a peering zone. If the zone is already deactivated, returns false for deactivate_succeeded field.

      Args:
        request: (DnsActivePeeringZonesDeactivateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PeeringZoneDeactivateResponse) The response message.
      """
    config = self.GetMethodConfig('Deactivate')
    return self._RunMethod(config, request, global_params=global_params)