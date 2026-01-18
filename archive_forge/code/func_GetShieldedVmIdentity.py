from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetShieldedVmIdentity(self, request, global_params=None):
    """Returns the Shielded VM Identity of an instance.

      Args:
        request: (ComputeInstancesGetShieldedVmIdentityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ShieldedVmIdentity) The response message.
      """
    config = self.GetMethodConfig('GetShieldedVmIdentity')
    return self._RunMethod(config, request, global_params=global_params)