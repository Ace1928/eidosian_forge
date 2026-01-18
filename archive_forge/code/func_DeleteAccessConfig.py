from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def DeleteAccessConfig(self, request, global_params=None):
    """Deletes an access config from an instance's network interface.

      Args:
        request: (ComputeInstancesDeleteAccessConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DeleteAccessConfig')
    return self._RunMethod(config, request, global_params=global_params)