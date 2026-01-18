from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetPrivateIpGoogleAccess(self, request, global_params=None):
    """Set whether VMs in this subnet can access Google services without assigning external IP addresses through Private Google Access.

      Args:
        request: (ComputeSubnetworksSetPrivateIpGoogleAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetPrivateIpGoogleAccess')
    return self._RunMethod(config, request, global_params=global_params)