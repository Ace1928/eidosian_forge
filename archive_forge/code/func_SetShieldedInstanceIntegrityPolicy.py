from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetShieldedInstanceIntegrityPolicy(self, request, global_params=None):
    """Sets the Shielded Instance integrity policy for an instance. You can only use this method on a running instance. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeInstancesSetShieldedInstanceIntegrityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetShieldedInstanceIntegrityPolicy')
    return self._RunMethod(config, request, global_params=global_params)