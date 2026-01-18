from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetNatMappingInfo(self, request, global_params=None):
    """Retrieves runtime Nat mapping information of VM endpoints.

      Args:
        request: (ComputeRoutersGetNatMappingInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VmEndpointNatMappingsList) The response message.
      """
    config = self.GetMethodConfig('GetNatMappingInfo')
    return self._RunMethod(config, request, global_params=global_params)