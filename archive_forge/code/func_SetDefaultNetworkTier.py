from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetDefaultNetworkTier(self, request, global_params=None):
    """Sets the default network tier of the project. The default network tier is used when an address/forwardingRule/instance is created without specifying the network tier field.

      Args:
        request: (ComputeProjectsSetDefaultNetworkTierRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetDefaultNetworkTier')
    return self._RunMethod(config, request, global_params=global_params)