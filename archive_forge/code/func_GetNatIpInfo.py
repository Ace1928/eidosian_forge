from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetNatIpInfo(self, request, global_params=None):
    """Retrieves runtime NAT IP information.

      Args:
        request: (ComputeRoutersGetNatIpInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NatIpInfoResponse) The response message.
      """
    config = self.GetMethodConfig('GetNatIpInfo')
    return self._RunMethod(config, request, global_params=global_params)